// MSM4 / MSM7 decoder kernel implementation. Single-pass walk of the
// header, masks, per-SV block, and per-cell signal block.

#include "msm_decode.hpp"

#include "bit_cursor.hpp"

#include <cmath>

namespace rinexpy_native {

namespace {

// Speed of light in m/ms — matches the _MSM_C constant in
// rinexpy.rtcm3 byte-for-byte (299792.458).
constexpr double SPEED_OF_LIGHT_M_PER_MS = 299792.458;

// 2^-29 and 2^-31 are exact in IEEE 754, so just compute them via
// std::ldexp once at compile time (constexpr ldexp isn't standard
// until C++23, so this is fine inline).
constexpr double MSM7_FINE_PR_SCALE    = 1.0 / (1ULL << 29);
constexpr double MSM7_FINE_PHASE_SCALE = 1.0 / (1ULL << 31);
constexpr double MSM4_FINE_PR_SCALE    = 1.0 / (1ULL << 24);
constexpr double MSM4_FINE_PHASE_SCALE = 1.0 / (1ULL << 29);

}  // namespace

MsmResult decode_msm(const std::uint8_t* body, std::size_t n_bytes,
                     int msm_kind) noexcept {
    MsmResult out;
    const std::size_t bits_total = n_bytes * 8;

    // Defensive: an MSM header needs 169 bits (12 ID + 12 sta_id + 30 tow
    // + 1 sync + 3 iod + 7 + 2 + 2 + 1 + 3 + 32 + 32 + 32). If we don't
    // have even that, the message is truncated; return an empty result.
    if (bits_total < 169) {
        out.payload_truncated = true;
        return out;
    }

    std::size_t bit = 12;  // past msg_id

    out.station_id = static_cast<int>(read_bits(body, n_bytes, bit, 12));
    bit += 12;
    out.tow_ms = static_cast<std::uint32_t>(read_bits(body, n_bytes, bit, 30));
    bit += 30;
    out.sync = static_cast<int>(read_bits(body, n_bytes, bit, 1));
    bit += 1;
    out.iod = static_cast<int>(read_bits(body, n_bytes, bit, 3));
    bit += 3;
    bit += 7;  // session time
    bit += 2;  // clock steering
    bit += 2;  // external clock
    out.smoothing_indicator = static_cast<int>(
        read_bits(body, n_bytes, bit, 1));
    bit += 1;
    out.smoothing_interval = static_cast<int>(
        read_bits(body, n_bytes, bit, 3));
    bit += 3;

    const std::uint64_t sv_mask_hi = read_bits(body, n_bytes, bit, 32);
    bit += 32;
    const std::uint64_t sv_mask_lo = read_bits(body, n_bytes, bit, 32);
    bit += 32;
    out.sv_mask = (sv_mask_hi << 32) | sv_mask_lo;

    out.signal_mask = static_cast<std::uint32_t>(
        read_bits(body, n_bytes, bit, 32));
    bit += 32;

    // Decompose masks. Same bit ordering as the Python: bit 63 of the
    // sv mask is SV 0, bit 31 of the signal mask is signal 0.
    out.sv_indices.reserve(64);
    for (int i = 0; i < 64; ++i) {
        if ((out.sv_mask >> (63 - i)) & 1ULL) {
            out.sv_indices.push_back(i);
        }
    }
    out.signal_indices.reserve(32);
    for (int i = 0; i < 32; ++i) {
        if ((out.signal_mask >> (31 - i)) & 1U) {
            out.signal_indices.push_back(i);
        }
    }
    out.n_sv = static_cast<int>(out.sv_indices.size());
    out.n_sig = static_cast<int>(out.signal_indices.size());

    // Cell mask: n_sv * n_sig bits.
    const std::size_t n_cells = static_cast<std::size_t>(out.n_sv) *
                                static_cast<std::size_t>(out.n_sig);
    if (bit + n_cells > bits_total) {
        out.payload_truncated = true;
        return out;
    }
    out.cell_mask.resize(n_cells);
    std::size_t n_present = 0;
    for (std::size_t i = 0; i < n_cells; ++i) {
        const std::uint8_t v = static_cast<std::uint8_t>(
            read_bits(body, n_bytes, bit + i, 1));
        out.cell_mask[i] = v;
        n_present += v;
    }
    bit += n_cells;

    // Per-satellite block layout (RTCM 10403.3 §3.5.16):
    //   MSM1, 2, 3:  rough_int_ms(8) + rough_mod_1ms(10)                  = 18
    //   MSM4, 6:     rough_int_ms(8) + ext_info(4) + rough_mod_1ms(10)    = 22
    //   MSM5, 7:     same as MSM4 + rough_doppler(14, signed)              = 36
    const bool has_ext_info       = (msm_kind == 4 || msm_kind == 5 ||
                                     msm_kind == 6 || msm_kind == 7);
    const bool has_rough_doppler  = (msm_kind == 5 || msm_kind == 7);
    const unsigned sat_block_bits = 8U
                                  + (has_ext_info ? 4U : 0U)
                                  + 10U
                                  + (has_rough_doppler ? 14U : 0U);
    if (bit + static_cast<std::size_t>(sat_block_bits) * out.n_sv > bits_total) {
        out.payload_truncated = true;
        return out;
    }
    // COLUMN-MAJOR sat block: read all SVs' rough_int_ms, then all
    // ext_info, then all rough_mod_1ms, then all rough_doppler.
    out.rough_range_ms.resize(out.n_sv);
    out.extended_info.resize(out.n_sv, 0);
    out.rough_doppler.resize(out.n_sv, 0);

    std::vector<std::uint32_t> rough_int_ms(out.n_sv);
    std::vector<std::uint32_t> rough_mod_1ms(out.n_sv);
    for (int k = 0; k < out.n_sv; ++k) {
        rough_int_ms[k] = static_cast<std::uint32_t>(
            read_bits(body, n_bytes, bit, 8));
        bit += 8;
    }
    if (has_ext_info) {
        for (int k = 0; k < out.n_sv; ++k) {
            out.extended_info[k] = static_cast<int>(
                read_bits(body, n_bytes, bit, 4));
            bit += 4;
        }
    }
    for (int k = 0; k < out.n_sv; ++k) {
        rough_mod_1ms[k] = static_cast<std::uint32_t>(
            read_bits(body, n_bytes, bit, 10));
        bit += 10;
    }
    if (has_rough_doppler) {
        for (int k = 0; k < out.n_sv; ++k) {
            out.rough_doppler[k] = static_cast<int>(
                read_bits_signed(body, n_bytes, bit, 14));
            bit += 14;
        }
    }
    for (int k = 0; k < out.n_sv; ++k) {
        out.rough_range_ms[k] = static_cast<double>(rough_int_ms[k])
                              + static_cast<double>(rough_mod_1ms[k]) / 1024.0;
    }

    // Per-cell signal block configuration:
    //   MSM1:  fine_pr(15s) + lock(4) + halfcyc(1) + cnr(6)                       = 26
    //   MSM2:  fine_phase(22s) + lock(4) + halfcyc(1) + cnr(6)                    = 33
    //   MSM3, MSM4: fine_pr(15s) + fine_phase(22s) + lock(4) + halfcyc(1) + cnr(6) = 48
    //   MSM5:  MSM4 layout + fine_doppler(15s)                                    = 63
    //   MSM6:  fine_pr(20s) + fine_phase(24s) + lock(10) + halfcyc(1) + cnr(10)   = 65
    //   MSM7:  MSM6 layout + fine_doppler(15s)                                    = 80
    const bool has_code             = (msm_kind != 2);
    const bool has_phase            = (msm_kind != 1);
    const bool hi_prec              = (msm_kind == 6 || msm_kind == 7);
    const bool has_fine_doppler     = (msm_kind == 5 || msm_kind == 7);
    const unsigned fine_pr_bits     = hi_prec ? 20U : 15U;
    const unsigned fine_phase_bits  = hi_prec ? 24U : 22U;
    const unsigned lock_bits        = hi_prec ? 10U : 4U;
    const unsigned cnr_bits         = hi_prec ? 10U : 6U;
    const double fine_pr_scale      = hi_prec ? MSM7_FINE_PR_SCALE    : MSM4_FINE_PR_SCALE;
    const double fine_phase_scale   = hi_prec ? MSM7_FINE_PHASE_SCALE : MSM4_FINE_PHASE_SCALE;
    const unsigned cell_bits        = (has_code ? fine_pr_bits : 0U)
                                    + (has_phase ? fine_phase_bits : 0U)
                                    + lock_bits + 1U + cnr_bits
                                    + (has_fine_doppler ? 15U : 0U);
    if (bit + static_cast<std::size_t>(cell_bits) * n_present > bits_total) {
        out.payload_truncated = true;
        return out;
    }

    out.obs_sv_k.reserve(n_present);
    out.obs_sig_k.reserve(n_present);
    out.pseudorange_m.reserve(n_present);
    out.phase_m.reserve(n_present);
    out.lock_time.reserve(n_present);
    out.half_cycle_ambiguity.reserve(n_present);
    out.cnr_dbhz.reserve(n_present);
    out.doppler_mps.reserve(n_present);

    const double nan_v = std::nan("");
    // COLUMN-MAJOR cell block: all cells' fine_pr, then fine_phase,
    // then lock, halfcyc, cnr, fine_doppler.
    std::vector<std::int64_t> fine_pr_v(n_present, 0);
    std::vector<std::int64_t> fine_phase_v(n_present, 0);
    std::vector<int> lock_v(n_present, 0);
    std::vector<int> halfcyc_v(n_present, 0);
    std::vector<std::uint64_t> cnr_raw_v(n_present, 0);
    std::vector<std::int64_t> fine_dop_v(n_present, 0);
    if (has_code) {
        for (std::size_t j = 0; j < n_present; ++j) {
            fine_pr_v[j] = read_bits_signed(body, n_bytes, bit, fine_pr_bits);
            bit += fine_pr_bits;
        }
    }
    if (has_phase) {
        for (std::size_t j = 0; j < n_present; ++j) {
            fine_phase_v[j] = read_bits_signed(body, n_bytes, bit, fine_phase_bits);
            bit += fine_phase_bits;
        }
    }
    for (std::size_t j = 0; j < n_present; ++j) {
        lock_v[j] = static_cast<int>(read_bits(body, n_bytes, bit, lock_bits));
        bit += lock_bits;
    }
    for (std::size_t j = 0; j < n_present; ++j) {
        halfcyc_v[j] = static_cast<int>(read_bits(body, n_bytes, bit, 1));
        bit += 1;
    }
    for (std::size_t j = 0; j < n_present; ++j) {
        cnr_raw_v[j] = read_bits(body, n_bytes, bit, cnr_bits);
        bit += cnr_bits;
    }
    if (has_fine_doppler) {
        for (std::size_t j = 0; j < n_present; ++j) {
            fine_dop_v[j] = read_bits_signed(body, n_bytes, bit, 15);
            bit += 15;
        }
    }

    out.obs_sv_k.reserve(n_present);
    out.obs_sig_k.reserve(n_present);
    out.pseudorange_m.reserve(n_present);
    out.phase_m.reserve(n_present);
    out.lock_time.reserve(n_present);
    out.half_cycle_ambiguity.reserve(n_present);
    out.cnr_dbhz.reserve(n_present);
    out.doppler_mps.reserve(n_present);

    std::size_t present_iter = 0;
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        if (!out.cell_mask[cell_idx]) continue;
        const int sv_k = static_cast<int>(cell_idx / out.n_sig);
        const int sig_k = static_cast<int>(cell_idx % out.n_sig);
        const double rough_ms = out.rough_range_ms[sv_k];
        const std::size_t j = present_iter++;

        const double pr_m = has_code
            ? (rough_ms + static_cast<double>(fine_pr_v[j]) * fine_pr_scale) * SPEED_OF_LIGHT_M_PER_MS
            : nan_v;
        const double phase_m = has_phase
            ? (rough_ms + static_cast<double>(fine_phase_v[j]) * fine_phase_scale) * SPEED_OF_LIGHT_M_PER_MS
            : nan_v;
        const double cnr = hi_prec
            ? (static_cast<double>(cnr_raw_v[j]) / 16.0)
            : static_cast<double>(cnr_raw_v[j]);
        const double doppler_mps_v = has_fine_doppler
            ? (static_cast<double>(fine_dop_v[j]) * 1e-4)
            : nan_v;

        out.obs_sv_k.push_back(sv_k);
        out.obs_sig_k.push_back(sig_k);
        out.pseudorange_m.push_back(pr_m);
        out.phase_m.push_back(phase_m);
        out.lock_time.push_back(lock_v[j]);
        out.half_cycle_ambiguity.push_back(halfcyc_v[j]);
        out.cnr_dbhz.push_back(cnr);
        out.doppler_mps.push_back(doppler_mps_v);
    }

    return out;
}

}  // namespace rinexpy_native
