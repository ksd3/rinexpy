// GPS LNAV + BeiDou D1/D2 subframe decoders.
//
// Both formats pack their navigation message bits into 30-bit "words"
// with 6 trailing parity bits (LNAV) or 4 / 8 trailing parity bits
// (BeiDou D1 word-1 / words-2..10). The Python references in
// rinexpy.gps_lnav and rinexpy.beidou build a Python *string* of
// '0'/'1' chars and then slice it per field; here we strip parity
// straight into a contiguous byte buffer and read each field via the
// bit_cursor.hpp helpers.
//
// The full subframe-decode body is one C++ call; the binding layer
// hands back a Python dict so callers (NAV3, RTCM3 1019/1042, the
// RXM-SFRBX pipeline) don't change.

#pragma once

#include <cstdint>
#include <vector>

namespace rinexpy_native {

// Result of decoding one LNAV subframe. Fields populated depend on
// subframe ID; ones not present default to 0.
struct LnavSubframe {
    int subframe_id = 0;
    int tow_count = 0;

    // Subframe 1 (clock + signal quality).
    int week = 0;
    int ca_or_p_on_l2 = 0;
    int ura = 0;
    int sv_health = 0;
    int iodc = 0;
    int l2_p_data_flag = 0;
    double tgd_s = 0.0;
    double toc_s = 0.0;
    double af0_s = 0.0;
    double af1_s_per_s = 0.0;
    double af2_s_per_s2 = 0.0;

    // Subframe 2 (ephemeris part 1).
    int iode2 = 0;
    double crs_m = 0.0;
    double delta_n_rad_s = 0.0;
    double m0_rad = 0.0;
    double cuc_rad = 0.0;
    double e_ = 0.0;
    double cus_rad = 0.0;
    double sqrt_a_root_m = 0.0;
    double toe_s = 0.0;
    int fit_interval_flag = 0;
    int aodo = 0;

    // Subframe 3 (ephemeris part 2).
    double cic_rad = 0.0;
    double omega0_rad = 0.0;
    double cis_rad = 0.0;
    double i0_rad = 0.0;
    double crc_m = 0.0;
    double omega_rad = 0.0;
    double omega_dot_rad_s = 0.0;
    int iode3 = 0;
    double idot_rad_s = 0.0;
};

// Decode one LNAV subframe. `words` is the 10 raw 30-bit ints
// (parity bits included; the kernel strips them). `expected_id` is
// 1, 2, or 3 — anything else raises (mirrors the Python check).
// Returns the populated struct; the binding wraps the appropriate
// subset into a dict.
LnavSubframe decode_lnav_subframe(const std::uint32_t* words,
                                  int expected_id);

// BeiDou D1 subframe 1 (clock + ionosphere) decoded fields.
struct BeidouD1Sf1 {
    int subframe_id = 1;
    int sow_s = 0;
    int sat_h1 = 0;
    int aodc = 0;
    int urai = 0;
    int week = 0;
    double toc_s = 0.0;
    double tgd1_s = 0.0;
    double tgd2_s = 0.0;
    double alpha0 = 0.0;
    double alpha1 = 0.0;
    double alpha2 = 0.0;
    double alpha3 = 0.0;
    double beta0 = 0.0;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double beta3 = 0.0;
    double a2_s_per_s2 = 0.0;
    double a0_s = 0.0;
    double a1_s_per_s = 0.0;
    int aode = 0;
};

// Decode a BeiDou D1 subframe 1. Validates the 11-bit preamble.
BeidouD1Sf1 decode_beidou_d1_sf1(const std::uint32_t* words);

// BeiDou D2 page 1 (clock parameters, paginated 500 bps GEO stream).
struct BeidouD2P1 {
    int subframe_id = 1;
    int page = 1;
    int sat_h1 = 0;
    int aodc = 0;
    int urai = 0;
    int week = 0;
    double toc_s = 0.0;
    double tgd1_s = 0.0;
    double tgd2_s = 0.0;
    double a0_s = 0.0;
    double a1_s_per_s = 0.0;
    double a2_s_per_s2 = 0.0;
    int aode = 0;
};

BeidouD2P1 decode_beidou_d2_page1(const std::uint32_t* words);

}  // namespace rinexpy_native
