// GPS LNAV + BeiDou D1/D2 subframe decoder implementations.
//
// Bit offsets and scale factors mirror rinexpy.gps_lnav and
// rinexpy.beidou exactly so values round-trip with the Python
// references.

#include "nav_subframes.hpp"

#include "bit_cursor.hpp"

#include <stdexcept>

namespace rinexpy_native {

namespace {

// ICD-GPS-200 Pi value (matches gps_lnav._PI bit-for-bit).
constexpr double PI_ICD = 3.1415926535898;

// LNAV preamble (1000 1011 = 0x8B).
constexpr std::uint32_t LNAV_PREAMBLE = 0x8B;

// BeiDou nav preamble (11 bits, 0x712).
constexpr std::uint32_t BEIDOU_PREAMBLE = 0x712;

// LNAV: pack the 10 30-bit words' high 24 data bits into a 240-bit
// (30-byte) MSB-first buffer. Each 24-bit chunk lives at bits
// w*24 .. (w+1)*24.
void pack_lnav_data(const std::uint32_t* words, std::uint8_t* out_30) noexcept {
    // Clear first.
    for (int i = 0; i < 30; ++i) out_30[i] = 0;
    // Each word contributes its high 24 bits = bits 6..29 of the
    // 30-bit raw int. Drop the 6 parity bits via >>6.
    for (int w = 0; w < 10; ++w) {
        const std::uint32_t v = (words[w] >> 6) & 0xFFFFFFU;
        // 24 bits land at bit-offset w*24 from the MSB of out_30.
        const int bit_off = w * 24;
        const int byte_off = bit_off / 8;
        // Aligned: w*24 is always a multiple of 8 (24, 48, 72, ...).
        out_30[byte_off]     = static_cast<std::uint8_t>((v >> 16) & 0xFF);
        out_30[byte_off + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
        out_30[byte_off + 2] = static_cast<std::uint8_t>(v & 0xFF);
    }
}

// BeiDou D1/D2: word 1 contributes its high 26 bits, words 2..10 each
// their high 22 bits. Total 26 + 9*22 = 224 bits = 28 bytes.
//
// Bit layout in the 224-bit MSB-first buffer:
//   word 1: bits 0..25
//   word 2: bits 26..47
//   word 3: bits 48..69
//   ...
//   word 10: bits 202..223
void pack_beidou_data(const std::uint32_t* words, std::uint8_t* out_28) noexcept {
    for (int i = 0; i < 28; ++i) out_28[i] = 0;

    // Word 1: high 26 bits (mask 0x3FFFFFF after >>4).
    const std::uint64_t w1 = (words[0] >> 4) & 0x3FFFFFFULL;
    // Write 26 bits at bit offset 0 of the 224-bit buffer (3 full bytes
    // plus 2 bits of byte 3).
    out_28[0] = static_cast<std::uint8_t>((w1 >> 18) & 0xFF);
    out_28[1] = static_cast<std::uint8_t>((w1 >> 10) & 0xFF);
    out_28[2] = static_cast<std::uint8_t>((w1 >> 2) & 0xFF);
    // Low 2 bits of w1 -> top 2 bits of out_28[3].
    out_28[3] = static_cast<std::uint8_t>((w1 & 0x3) << 6);

    // Words 2..10: each contributes high 22 bits (mask 0x3FFFFF after >>8).
    // The destination bit offset starts at 26 and grows by 22 per word.
    int dst_bit = 26;
    for (int w = 1; w < 10; ++w) {
        const std::uint64_t dv = (words[w] >> 8) & 0x3FFFFFULL;
        // Write 22 bits at MSB-aligned bit-offset dst_bit.
        for (int b = 0; b < 22; ++b) {
            const int abs_bit = dst_bit + b;
            const int byte_idx = abs_bit >> 3;
            const int bit_in_byte = 7 - (abs_bit & 7);
            const std::uint8_t bit_v = static_cast<std::uint8_t>(
                (dv >> (21 - b)) & 1ULL);
            out_28[byte_idx] |= static_cast<std::uint8_t>(bit_v << bit_in_byte);
        }
        dst_bit += 22;
    }
}

// Convenience: 64-bit unsigned read on a fixed-size buffer.
inline std::uint64_t U(const std::uint8_t* buf, std::size_t n, std::size_t s, unsigned k) {
    return read_bits(buf, n, s, k);
}
inline std::int64_t S(const std::uint8_t* buf, std::size_t n, std::size_t s, unsigned k) {
    return read_bits_signed(buf, n, s, k);
}

}  // namespace

LnavSubframe decode_lnav_subframe(const std::uint32_t* words, int expected_id) {
    if (expected_id < 1 || expected_id > 3) {
        throw std::invalid_argument(
            "decode_lnav_subframe: expected_id must be 1, 2, or 3");
    }
    std::uint8_t buf[30];
    pack_lnav_data(words, buf);
    constexpr std::size_t N = 30;

    const std::uint64_t pre = U(buf, N, 0, 8);
    if (pre != LNAV_PREAMBLE) {
        throw std::invalid_argument("bad GPS LNAV preamble");
    }
    const std::uint64_t sf = U(buf, N, 43, 3);
    if (static_cast<int>(sf) != expected_id) {
        throw std::invalid_argument("LNAV subframe id mismatch");
    }

    LnavSubframe out;
    out.subframe_id = static_cast<int>(sf);
    out.tow_count = static_cast<int>(U(buf, N, 24, 17));

    if (expected_id == 1) {
        out.week = static_cast<int>(U(buf, N, 48, 10));
        out.ca_or_p_on_l2 = static_cast<int>(U(buf, N, 58, 2));
        out.ura = static_cast<int>(U(buf, N, 60, 4));
        out.sv_health = static_cast<int>(U(buf, N, 64, 6));
        const int iodc_msb = static_cast<int>(U(buf, N, 70, 2));
        out.l2_p_data_flag = static_cast<int>(U(buf, N, 72, 1));
        out.tgd_s = static_cast<double>(S(buf, N, 160, 8)) * 0x1p-31;
        const int iodc_lsb = static_cast<int>(U(buf, N, 168, 8));
        out.iodc = (iodc_msb << 8) | iodc_lsb;
        out.toc_s = static_cast<double>(U(buf, N, 176, 16)) * 16.0;
        out.af2_s_per_s2 = static_cast<double>(S(buf, N, 192, 8)) * 0x1p-55;
        out.af1_s_per_s = static_cast<double>(S(buf, N, 200, 16)) * 0x1p-43;
        out.af0_s = static_cast<double>(S(buf, N, 216, 22)) * 0x1p-31;
    } else if (expected_id == 2) {
        out.iode2 = static_cast<int>(U(buf, N, 48, 8));
        out.crs_m = static_cast<double>(S(buf, N, 56, 16)) * 0x1p-5;
        const std::int64_t dn_sc = S(buf, N, 72, 16);
        out.delta_n_rad_s = static_cast<double>(dn_sc) * 0x1p-43 * PI_ICD;
        // M_0: 32-bit signed split across (88, 8) MSB and (96, 24) LSB.
        const std::uint64_t m0_hi = U(buf, N, 88, 8);
        const std::uint64_t m0_lo = U(buf, N, 96, 24);
        std::int64_t m0_u = static_cast<std::int64_t>((m0_hi << 24) | m0_lo);
        if (m0_u & (1LL << 31)) m0_u -= (1LL << 32);
        out.m0_rad = static_cast<double>(m0_u) * 0x1p-31 * PI_ICD;
        out.cuc_rad = static_cast<double>(S(buf, N, 120, 16)) * 0x1p-29;
        // e: 32-bit unsigned split (136, 8) high + (144, 24) low.
        const std::uint64_t e_u = (U(buf, N, 136, 8) << 24) | U(buf, N, 144, 24);
        out.e_ = static_cast<double>(e_u) * 0x1p-33;
        out.cus_rad = static_cast<double>(S(buf, N, 168, 16)) * 0x1p-29;
        // sqrt(A): 32-bit unsigned split (184, 8) high + (192, 24) low.
        const std::uint64_t sa = (U(buf, N, 184, 8) << 24) | U(buf, N, 192, 24);
        out.sqrt_a_root_m = static_cast<double>(sa) * 0x1p-19;
        out.toe_s = static_cast<double>(U(buf, N, 216, 16)) * 16.0;
        out.fit_interval_flag = static_cast<int>(U(buf, N, 232, 1));
        out.aodo = static_cast<int>(U(buf, N, 233, 5));
    } else {  // subframe 3
        out.cic_rad = static_cast<double>(S(buf, N, 48, 16)) * 0x1p-29;
        // Omega_0: 32-bit signed split (64, 8) + (72, 24).
        const std::uint64_t o0_hi = U(buf, N, 64, 8);
        const std::uint64_t o0_lo = U(buf, N, 72, 24);
        std::int64_t o0 = static_cast<std::int64_t>((o0_hi << 24) | o0_lo);
        if (o0 & (1LL << 31)) o0 -= (1LL << 32);
        out.omega0_rad = static_cast<double>(o0) * 0x1p-31 * PI_ICD;
        out.cis_rad = static_cast<double>(S(buf, N, 96, 16)) * 0x1p-29;
        // i_0: 32-bit signed split (112, 8) + (120, 24).
        const std::uint64_t i_hi = U(buf, N, 112, 8);
        const std::uint64_t i_lo = U(buf, N, 120, 24);
        std::int64_t i0_u = static_cast<std::int64_t>((i_hi << 24) | i_lo);
        if (i0_u & (1LL << 31)) i0_u -= (1LL << 32);
        out.i0_rad = static_cast<double>(i0_u) * 0x1p-31 * PI_ICD;
        out.crc_m = static_cast<double>(S(buf, N, 144, 16)) * 0x1p-5;
        // omega: 32-bit signed split (160, 8) + (168, 24).
        const std::uint64_t om_hi = U(buf, N, 160, 8);
        const std::uint64_t om_lo = U(buf, N, 168, 24);
        std::int64_t om = static_cast<std::int64_t>((om_hi << 24) | om_lo);
        if (om & (1LL << 31)) om -= (1LL << 32);
        out.omega_rad = static_cast<double>(om) * 0x1p-31 * PI_ICD;
        out.omega_dot_rad_s = static_cast<double>(S(buf, N, 192, 24)) *
                              0x1p-43 * PI_ICD;
        out.iode3 = static_cast<int>(U(buf, N, 216, 8));
        out.idot_rad_s = static_cast<double>(S(buf, N, 224, 14)) *
                          0x1p-43 * PI_ICD;
    }
    return out;
}

BeidouD1Sf1 decode_beidou_d1_sf1(const std::uint32_t* words) {
    std::uint8_t buf[28];
    pack_beidou_data(words, buf);
    constexpr std::size_t N = 28;

    const std::uint64_t pre = U(buf, N, 0, 11);
    if (pre != BEIDOU_PREAMBLE) {
        throw std::invalid_argument("bad BeiDou preamble");
    }
    const std::uint64_t fra_id = U(buf, N, 23, 3);
    if (fra_id != 1) {
        throw std::invalid_argument("expected BeiDou D1 subframe 1");
    }

    BeidouD1Sf1 out;
    out.sat_h1 = static_cast<int>(U(buf, N, 38, 1));
    out.aodc = static_cast<int>(U(buf, N, 39, 5));
    out.urai = static_cast<int>(U(buf, N, 44, 4));
    out.week = static_cast<int>(U(buf, N, 48, 13));
    out.toc_s = static_cast<double>(U(buf, N, 61, 17)) * 8.0;
    out.tgd1_s = static_cast<double>(S(buf, N, 78, 10)) * 0.1e-9;
    out.tgd2_s = static_cast<double>(S(buf, N, 88, 10)) * 0.1e-9;
    out.alpha0 = static_cast<double>(S(buf, N,  98, 8)) * 0x1p-30;
    out.alpha1 = static_cast<double>(S(buf, N, 106, 8)) * 0x1p-27;
    out.alpha2 = static_cast<double>(S(buf, N, 114, 8)) * 0x1p-24;
    out.alpha3 = static_cast<double>(S(buf, N, 122, 8)) * 0x1p-24;
    out.beta0 = static_cast<double>(S(buf, N, 130, 8)) * static_cast<double>(1 << 11);
    out.beta1 = static_cast<double>(S(buf, N, 138, 8)) * static_cast<double>(1 << 14);
    out.beta2 = static_cast<double>(S(buf, N, 146, 8)) * static_cast<double>(1 << 16);
    out.beta3 = static_cast<double>(S(buf, N, 154, 8)) * static_cast<double>(1 << 16);
    out.a2_s_per_s2 = static_cast<double>(S(buf, N, 162, 11)) * 0x1p-66;
    out.a0_s = static_cast<double>(S(buf, N, 173, 24)) * 0x1p-33;
    out.a1_s_per_s = static_cast<double>(S(buf, N, 197, 22)) * 0x1p-50;
    out.aode = static_cast<int>(U(buf, N, 219, 5));
    return out;
}

BeidouD2P1 decode_beidou_d2_page1(const std::uint32_t* words) {
    std::uint8_t buf[28];
    pack_beidou_data(words, buf);
    constexpr std::size_t N = 28;

    const std::uint64_t pre = U(buf, N, 0, 11);
    if (pre != BEIDOU_PREAMBLE) {
        throw std::invalid_argument("bad BeiDou preamble");
    }
    const std::uint64_t fra_id = U(buf, N, 23, 3);
    const std::uint64_t page = U(buf, N, 38, 4);
    if (fra_id != 1 || page != 1) {
        throw std::invalid_argument("expected BeiDou D2 frame=1 page=1");
    }

    BeidouD2P1 out;
    out.sat_h1 = static_cast<int>(U(buf, N, 42, 1));
    out.aodc = static_cast<int>(U(buf, N, 43, 5));
    out.urai = static_cast<int>(U(buf, N, 48, 4));
    out.week = static_cast<int>(U(buf, N, 52, 13));
    out.toc_s = static_cast<double>(U(buf, N, 65, 17)) * 8.0;
    out.tgd1_s = static_cast<double>(S(buf, N, 82, 10)) * 0.1e-9;
    out.tgd2_s = static_cast<double>(S(buf, N, 92, 10)) * 0.1e-9;
    out.a0_s = static_cast<double>(S(buf, N, 102, 24)) * 0x1p-33;
    out.a1_s_per_s = static_cast<double>(S(buf, N, 126, 22)) * 0x1p-50;
    out.a2_s_per_s2 = static_cast<double>(S(buf, N, 148, 11)) * 0x1p-66;
    out.aode = static_cast<int>(U(buf, N, 159, 5));
    return out;
}

}  // namespace rinexpy_native
