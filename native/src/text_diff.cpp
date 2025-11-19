// CRINEX TextDiff kernel implementation.

#include "text_diff.hpp"

namespace rinexpy_native {

void TextDiffState::reset() noexcept {
    ref_.clear();
}

std::string TextDiffState::step(const std::string& delta) {
    // Leading '&' at position 0 means "this line is an absolute
    // reinit; ignore prior state".
    if (!delta.empty() && delta[0] == '&') {
        ref_ = delta.substr(1);
        // Convert any literal '&' inside the reinit body back to space
        // -- the reinit form follows the same '&' = space convention.
        for (char& c : ref_) {
            if (c == '&') c = ' ';
        }
        return ref_;
    }

    // Position-wise delta. The output length equals the delta length.
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
