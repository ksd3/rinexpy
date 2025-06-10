// MSM4 / MSM7 frame decoder kernel.
//
// Maps directly onto rinexpy.rtcm3._decode_msm_header: parses the
// header, walks the SV mask + signal mask + cell mask, decodes the
// per-satellite block, and decodes the per-cell signal block. The
// public entry point returns a flat result struct of parallel arrays;
// the binding layer converts that to NumPy arrays and the Python
// wrapper reassembles the dict-of-list-of-dicts shape that
// rinexpy.rtcm3 currently exposes.
//
// Numerical contract matches the Python reference bit-for-bit:
//
// - Same SI conversions for pseudorange / phase / Doppler.
// - Same fine-PR / fine-phase scale factors (2^-29 / 2^-31 for MSM7,
//   2^-24 / 2^-29 for MSM4).
// - Same `rough_range_ms = rough_int_ms + rough_mod_1ms / 1024`.
// - Same `cnr / 16.0` for MSM7, raw integer cast to float for MSM4.
// - Truncated payloads set `payload_truncated = true` and leave the
//   per-SV / per-cell vectors empty (matching the Python early return).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rinexpy_native {

struct MsmResult {
    // Header fields (always populated).
    int station_id = 0;
    std::uint32_t tow_ms = 0;
    int sync = 0;
    int iod = 0;
    int smoothing_indicator = 0;
    int smoothing_interval = 0;
    std::uint64_t sv_mask = 0;
    std::uint32_t signal_mask = 0;
    int n_sv = 0;
    int n_sig = 0;

    // SV / signal index decompositions (sizes n_sv / n_sig).
    std::vector<int> sv_indices;
    std::vector<int> signal_indices;

    // Cell-mask bits (size n_sv * n_sig), 0/1. Empty if truncated.
    std::vector<std::uint8_t> cell_mask;

    // Per-satellite block (size n_sv each).
    std::vector<double> rough_range_ms;
    std::vector<int> extended_info;
    std::vector<int> rough_doppler;   // raw signed int from the wire

    // Per-cell observations (size = number of set bits in cell_mask).
    // obs_sv_k / obs_sig_k are indices into sv_indices / signal_indices.
    std::vector<int> obs_sv_k;
    std::vector<int> obs_sig_k;
    std::vector<double> pseudorange_m;
    std::vector<double> phase_m;
    std::vector<int> lock_time;
    std::vector<int> half_cycle_ambiguity;
    std::vector<double> cnr_dbhz;
    std::vector<double> doppler_mps;

    bool payload_truncated = false;
};

// Decode an MSM frame body (the bytes after the preamble + length but
// before the CRC). `msg_id` is forwarded as-is for the caller's
// records; this kernel branches solely on `msm_kind` (4 or 7).
MsmResult decode_msm(const std::uint8_t* body, std::size_t n_bytes,
                     int msm_kind) noexcept;

}  // namespace rinexpy_native
