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
    // sv_label: 3-char PRN (e.g. "G07", "C08"). For RINEX 3 output
    // the caller can prepend `sv_label` to the returned obs string;
    // for RINEX 2 the PRN isn't on the obs line so the label is
    // metadata only. n_obs: number of obs types declared for this
    // SV. flags_are_absolute: CRINEX 1 transmits the LLI/SSI string
    // as absolute text per epoch (rstrip of trailing spaces). CRINEX
    // 3 transmits it as a positional TextDiff. Default false (= v3
    // semantics) preserves the original behavior.
    CrinexSVDecoder(std::string sv_label, std::size_t n_obs,
                    bool flags_are_absolute = false);

    // Decode one CRINEX data line. Returns the raw n_obs * 16-char
    // observation string (F14.3 value + LLI + SSI per obs), no PRN
    // prefix, no rstrip. The caller is responsible for emitting it
    // in the right RINEX-version format (single line with PRN
    // for v3; wrapped at 5 obs per line for v2).
    std::string decode_line(const std::string& crx_line);

    // The PRN this decoder is associated with (returned to callers
    // that need to thread it through metadata).
    const std::string& sv_label() const noexcept { return sv_; }

private:
    std::string sv_;
    std::size_t n_obs_;
    bool flags_absolute_;
    std::vector<CrinexChannelState> channels_;
    std::vector<std::uint8_t> filled_;  // bool flags per channel
    TextDiffState flags_;           // CRINEX 3 mode
    std::string flags_abs_state_;   // CRINEX 1 mode: literal prev LLI/SSI
};

}  // namespace rinexpy_native
