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

    // Per-satellite block: 8+4+10+14 bits per SV (= 36 each).
    if (bit + 36ULL * out.n_sv > bits_total) {
        out.payload_truncated = true;
        return out;
    }
    out.rough_range_ms.resize(out.n_sv);
    out.extended_info.resize(out.n_sv);
    out.rough_doppler.resize(out.n_sv);
    for (int k = 0; k < out.n_sv; ++k) {
        const std::uint64_t rough_int_ms = read_bits(body, n_bytes, bit, 8);
        bit += 8;
        const std::uint64_t ext_info = read_bits(body, n_bytes, bit, 4);
        bit += 4;
        const std::uint64_t rough_mod_1ms = read_bits(body, n_bytes, bit, 10);
        bit += 10;
        const std::int64_t rough_doppler = read_bits_signed(
            body, n_bytes, bit, 14);
        bit += 14;
        out.rough_range_ms[k] = static_cast<double>(rough_int_ms)
                              + static_cast<double>(rough_mod_1ms) / 1024.0;
        out.extended_info[k] = static_cast<int>(ext_info);
        out.rough_doppler[k] = static_cast<int>(rough_doppler);
    }

    // Per-cell signal block.
    const unsigned bits_per_cell = (msm_kind == 7) ? 80U : 48U;
    if (bit + static_cast<std::size_t>(bits_per_cell) * n_present > bits_total) {
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
    for (std::size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        if (!out.cell_mask[cell_idx]) continue;

        const int sv_k = static_cast<int>(cell_idx / out.n_sig);
        const int sig_k = static_cast<int>(cell_idx % out.n_sig);
        const double rough_ms = out.rough_range_ms[sv_k];

        double pr_m, phase_m, doppler_mps_v, cnr;
        int lock, halfcyc;

        if (msm_kind == 7) {
            const std::int64_t fine_pr = read_bits_signed(
                body, n_bytes, bit, 20);
            bit += 20;
            const std::int64_t fine_phase = read_bits_signed(
                body, n_bytes, bit, 24);
            bit += 24;
            lock = static_cast<int>(read_bits(body, n_bytes, bit, 10));
            bit += 10;
            halfcyc = static_cast<int>(read_bits(body, n_bytes, bit, 1));
            bit += 1;
            cnr = static_cast<double>(read_bits(body, n_bytes, bit, 10))
                  / 16.0;
            bit += 10;
            const std::int64_t fine_dop = read_bits_signed(
                body, n_bytes, bit, 15);
            bit += 15;
            pr_m = (rough_ms + static_cast<double>(fine_pr)
                    * MSM7_FINE_PR_SCALE) * SPEED_OF_LIGHT_M_PER_MS;
            phase_m = (rough_ms + static_cast<double>(fine_phase)
                       * MSM7_FINE_PHASE_SCALE) * SPEED_OF_LIGHT_M_PER_MS;
            doppler_mps_v = static_cast<double>(fine_dop) * 1e-4;
        } else {
            const std::int64_t fine_pr = read_bits_signed(
                body, n_bytes, bit, 15);
            bit += 15;
            const std::int64_t fine_phase = read_bits_signed(
                body, n_bytes, bit, 22);
            bit += 22;
            lock = static_cast<int>(read_bits(body, n_bytes, bit, 4));
            bit += 4;
            halfcyc = static_cast<int>(read_bits(body, n_bytes, bit, 1));
            bit += 1;
            cnr = static_cast<double>(read_bits(body, n_bytes, bit, 6));
            bit += 6;
            pr_m = (rough_ms + static_cast<double>(fine_pr)
                    * MSM4_FINE_PR_SCALE) * SPEED_OF_LIGHT_M_PER_MS;
            phase_m = (rough_ms + static_cast<double>(fine_phase)
                       * MSM4_FINE_PHASE_SCALE) * SPEED_OF_LIGHT_M_PER_MS;
            doppler_mps_v = nan_v;
        }

        out.obs_sv_k.push_back(sv_k);
        out.obs_sig_k.push_back(sig_k);
        out.pseudorange_m.push_back(pr_m);
        out.phase_m.push_back(phase_m);
        out.lock_time.push_back(lock);
        out.half_cycle_ambiguity.push_back(halfcyc);
        out.cnr_dbhz.push_back(cnr);
        out.doppler_mps.push_back(doppler_mps_v);
    }

    return out;
}

}  // namespace rinexpy_native
