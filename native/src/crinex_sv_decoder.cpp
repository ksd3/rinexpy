// Fused per-SV CRINEX data-line decoder implementation.

#include "crinex_sv_decoder.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rinexpy_native {

namespace {

// Parse a signed integer from [begin, end) without allocating a
// std::string. Returns the value; sets *ok to false on parse error
// (only if there are non-digit / non-sign characters).
inline std::int64_t parse_int(const char* begin, const char* end, bool* ok) noexcept {
    *ok = true;
    if (begin == end) {
        *ok = false;
        return 0;
    }
    std::int64_t sign = 1;
    if (*begin == '-') { sign = -1; ++begin; }
    else if (*begin == '+') { ++begin; }
    std::int64_t v = 0;
    for (; begin != end; ++begin) {
        const char c = *begin;
        if (c < '0' || c > '9') {
            *ok = false;
            return 0;
        }
        v = v * 10 + (c - '0');
    }
    return sign * v;
}

// Format `int_value` (= value_m * 1000) into a 14-char RINEX 3 F14.3
// field. Writes 14 characters into out[0..13]. Matches the hatanaka
// reference: when |value| < 1, the leading "0" is omitted ("0.192"
// becomes ".192", "-0.192" becomes "-.192").
void format_f14_3(std::int64_t v, char* out) noexcept {
    char body[24];
    int body_len = 0;
    char sign = '\0';
    std::int64_t absv = v;
    if (v < 0) {
        sign = '-';
        absv = -v;
    }
    const std::int64_t whole = absv / 1000;
    const std::int64_t frac = absv % 1000;
    if (whole == 0) {
        // ".xxx" or "-.xxx"
        if (sign) {
            body[body_len++] = sign;
        }
        body[body_len++] = '.';
        body[body_len++] = static_cast<char>('0' + (frac / 100) % 10);
        body[body_len++] = static_cast<char>('0' + (frac / 10) % 10);
        body[body_len++] = static_cast<char>('0' + frac % 10);
    } else {
        char wb[24];
        int wb_len = 0;
        std::int64_t w = whole;
        while (w > 0) {
            wb[wb_len++] = static_cast<char>('0' + (w % 10));
            w /= 10;
        }
        if (sign) {
            body[body_len++] = sign;
        }
        // Reverse the digits into body.
        for (int i = wb_len - 1; i >= 0; --i) {
            body[body_len++] = wb[i];
        }
        body[body_len++] = '.';
        body[body_len++] = static_cast<char>('0' + (frac / 100) % 10);
        body[body_len++] = static_cast<char>('0' + (frac / 10) % 10);
        body[body_len++] = static_cast<char>('0' + frac % 10);
    }
    // Right-justify into 14 chars (pad left with spaces).
    const int pad = 14 - body_len;
    for (int i = 0; i < pad; ++i) {
        out[i] = ' ';
    }
    for (int i = 0; i < body_len; ++i) {
        out[pad + i] = body[i];
    }
}

}  // namespace

CrinexSVDecoder::CrinexSVDecoder(std::string sv_label, std::size_t n_obs,
                                 bool flags_are_absolute)
    : sv_(std::move(sv_label)),
      n_obs_(n_obs),
      flags_absolute_(flags_are_absolute),
      channels_(n_obs),
      filled_(n_obs, 0),
      flags_abs_state_(2 * n_obs, ' ') {
    for (auto& c : channels_) {
        crinex_channel_reset(&c, 0);
    }
}

std::string CrinexSVDecoder::decode_line(const std::string& line) {
    // Output buffer: n_obs * 16. Caller prepends a 3-char PRN for
    // RINEX 3 output, or wraps the buffer into 80-char (5-obs) lines
    // for RINEX 2 output. No rstrip; trailing-space trimming is
    // version-specific.
    std::string out(n_obs_ * 16, ' ');

    const char* p = line.data();
    const char* end = p + line.size();

    for (std::size_t i = 0; i < n_obs_; ++i) {
        // Find next space (or end).
        const char* sp = p;
        while (sp < end && *sp != ' ') ++sp;
        if (sp == p) {
            // Empty token -> obs missing at this epoch. Reset state
            // so the next non-empty token has to carry a "k&" init.
            crinex_channel_reset(&channels_[i], 0);
            filled_[i] = 0;
            // Out buffer already prefilled with spaces.
        } else {
            // Non-empty: parse it.
            // Look for "k&" prefix where k is a single digit.
            std::int64_t value;
            if ((sp - p) >= 3 && p[1] == '&' && p[0] >= '0' && p[0] <= '9') {
                const int order = p[0] - '0';
                bool ok = false;
                value = parse_int(p + 2, sp, &ok);
                if (ok) {
                    crinex_channel_reset(&channels_[i], order);
                    value = crinex_channel_step(&channels_[i], value);
                    filled_[i] = 1;
                } else {
                    // Malformed init token; skip.
                    filled_[i] = 0;
                }
            } else {
                bool ok = false;
                std::int64_t delta = parse_int(p, sp, &ok);
                if (!ok) {
                    // Malformed; skip.
                } else if (!filled_[i]) {
                    // No init yet but got a plain delta: use default
                    // order (3) and treat this as init.
                    crinex_channel_reset(&channels_[i], 3);
                    value = crinex_channel_step(&channels_[i], delta);
                    filled_[i] = 1;
                } else {
                    value = crinex_channel_step(&channels_[i], delta);
                }
            }
            if (filled_[i]) {
                format_f14_3(value, &out[i * 16]);
            }
        }
        // Advance p past the space (if any).
        p = (sp < end) ? sp + 1 : end;
    }

    // Whatever remains is the LLI/SSI text.
    //
    // CRINEX 3: positional TextDiff against the previous epoch
    //   (' ' = keep prev, '&' = the new char is space, others = override).
    //
    // CRINEX 1: literal override at every position the encoder
    //   transmitted, keep prev at trailing positions the encoder
    //   omitted (rstripped). To express "this position is space"
    //   the encoder just writes a literal ' '. Empty input means
    //   the entire LLI/SSI is unchanged from the previous epoch.
    std::string flag_text;
    if (flags_absolute_) {
        // CRINEX 1 uses pair-level TextDiff on the LLI/SSI string:
        // each obs has a 2-char (LLI, SSI) pair. A pair of two spaces
        // in the input means "keep the previous epoch's pair";
        // anything else (one or both chars non-space) overrides the
        // pair atomically. The encoder rstrips trailing all-space
        // pairs, so positions past the input length also keep prev.
        const std::size_t in_pairs = static_cast<std::size_t>(end - p) / 2;
        for (std::size_t k = 0; k < in_pairs && k < n_obs_; ++k) {
            const char a = p[2 * k];
            const char b = p[2 * k + 1];
            if (a == ' ' && b == ' ') continue;  // keep prev pair
            flags_abs_state_[2 * k]     = a;
            flags_abs_state_[2 * k + 1] = b;
        }
        flag_text = flags_abs_state_;
    } else {
        std::string flag_delta(p, end - p);
        flag_delta.resize(2 * n_obs_, ' ');
        flag_text = flags_.step(flag_delta);
    }
    // Place flag chars at offsets 14, 15 of each obs. For obs that
    // are missing at this epoch (filled_[i] == 0), the entire 16-char
    // slot stays blank regardless of any LLI/SSI state — the encoder
    // doesn't carry flag values for absent observations.
    for (std::size_t i = 0; i < n_obs_; ++i) {
        if (!filled_[i]) continue;
        const std::size_t base = i * 16 + 14;
        out[base]     = (2 * i     < flag_text.size()) ? flag_text[2 * i] : ' ';
        out[base + 1] = (2 * i + 1 < flag_text.size()) ? flag_text[2 * i + 1] : ' ';
    }
    return out;
}

}  // namespace rinexpy_native
