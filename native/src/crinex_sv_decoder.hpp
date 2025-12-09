// Fused per-SV CRINEX data-line decoder.
//
// The pure-Python orchestration in rinexpy.crinex was profiling at
// ~3 s on the 18 MB CEBR00ESP fixture, vs ~120 ms for the reference
// C tool. Per-call FFI overhead through individual TextDiff /
// CrinexChannel step() calls plus Python-side tokenisation,
// regex matching, integer formatting, and list assembly made up
// most of that 25x gap.
//
// CrinexSVDecoder fuses the whole per-SV step into one C++ call:
//   - tokenises the data line in place,
//   - runs each numeric token through the per-obs differencing state,
//   - formats reconstructed integers as RINEX 3 F14.3 (with the
//     hatanaka leading-zero-omission convention),
//   - feeds the trailing LLI/SSI string through TextDiff,
//   - assembles the standard RINEX 3 obs line ("PRN" + n_obs*16 chars).
//
// The Python wrapper just keeps one CrinexSVDecoder per SV and calls
// .decode_line() per epoch.

#pragma once

#include "crinex_diff.hpp"
#include "text_diff.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace rinexpy_native {

class CrinexSVDecoder {
public:
    // sv_label: 3-char PRN (e.g. "G07", "C08"). n_obs: number of obs
    // types declared for this SV's constellation.
    CrinexSVDecoder(std::string sv_label, std::size_t n_obs);

    // Decode one CRINEX data line. Returns the standard RINEX 3 obs
    // line for this epoch (PRN + n_obs * 16-char obs fields). Trailing
    // whitespace is rstripped to match hatanaka's output.
    std::string decode_line(const std::string& crx_line);

private:
    std::string sv_;
    std::size_t n_obs_;
    std::vector<CrinexChannelState> channels_;
    std::vector<std::uint8_t> filled_;  // bool flags per channel
    TextDiffState flags_;
};

}  // namespace rinexpy_native
