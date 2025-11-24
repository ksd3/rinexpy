// CRINEX TextDiff kernel implementation.

#include "text_diff.hpp"

namespace rinexpy_native {

void TextDiffState::reset() noexcept {
    ref_.clear();
}

std::string TextDiffState::step(const std::string& delta) {
    // Position-wise delta. The output length equals the delta length.
    // Empirically the leading-'&' special-case (mentioned in some
    // descriptions of the format as a reinit marker) does not apply
    // here: real CRINEX streams use '&' uniformly to mean "the new
    // character at this position is ' '". Caller can drive a fresh
    // initialisation by calling reset() before the first step().
    const std::size_t n = delta.size();
    std::string out(n, ' ');
    for (std::size_t i = 0; i < n; ++i) {
        const char d = delta[i];
        if (d == ' ') {
            // Unchanged: pull char from the reference (or space if past end).
            out[i] = (i < ref_.size()) ? ref_[i] : ' ';
        } else if (d == '&') {
            out[i] = ' ';
        } else {
            out[i] = d;
        }
    }
    ref_ = out;
    return out;
}

}  // namespace rinexpy_native
