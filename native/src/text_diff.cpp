// CRINEX TextDiff kernel implementation.

#include "text_diff.hpp"

namespace rinexpy_native {

void TextDiffState::reset() noexcept {
    ref_.clear();
}

std::string TextDiffState::step(const std::string& delta) {
    // Position-wise delta. The output length is max(delta_len, ref_len):
    // positions covered by the delta are overlaid; positions past the
    // end of the delta inherit from the reference (CRINEX encoders
    // omit unchanged trailing characters on epoch lines that change
    // only their leading content).
    //
    // Empirically the leading-'&' special-case (mentioned in some
    // descriptions of the format as a reinit marker) does not apply
    // here: real CRINEX streams use '&' uniformly to mean "the new
    // character at this position is ' '". Caller can drive a fresh
    // initialisation by calling reset() before the first step().
    const std::size_t dn = delta.size();
    const std::size_t rn = ref_.size();
    const std::size_t n = dn > rn ? dn : rn;
    std::string out(n, ' ');
    for (std::size_t i = 0; i < n; ++i) {
        if (i < dn) {
            const char d = delta[i];
            if (d == ' ') {
                out[i] = (i < rn) ? ref_[i] : ' ';
            } else if (d == '&') {
                out[i] = ' ';
            } else {
                out[i] = d;
            }
        } else {
            out[i] = ref_[i];
        }
    }
    ref_ = out;
    return out;
}

}  // namespace rinexpy_native
