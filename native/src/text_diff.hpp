// CRINEX TextDiff (character-wise delta) primitive.
//
// Reference: Hatanaka, Y. (2008), §2. Each text-field line in CRINEX
// is encoded as a position-wise delta against the previous decoded
// line:
//
//   - ' ' (space)  at position i  =>  keep the previous char at i
//   - '&' (ampersand) at position i =>  the new char at i is ' ' (space)
//   - any other char c at position i =>  the new char at i is c
//
// When the input is longer than the current reference, the reference
// is extended with spaces. When the input is shorter, the result has
// the input's length (trailing chars of the reference are dropped).
//
// A leading '&' at position 0 of an input line is special: it means
// "reinit; the rest of the line (from position 1 onward) is the
// absolute text". The decoder replaces the saved state with that.

#pragma once

#include <cstddef>
#include <string>

namespace rinexpy_native {

class TextDiffState {
public:
    // Reset to an empty reference. Subsequent step() calls must be
    // either '&'-prefixed initializations or pure-space-padded deltas.
    void reset() noexcept;

    // Apply one input delta line and return the reconstructed
    // absolute line. The caller owns the returned string.
    std::string step(const std::string& delta);

    // Inspect current reference text (used by tests).
    const std::string& current() const noexcept { return ref_; }

private:
    std::string ref_;
};

}  // namespace rinexpy_native
