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

CrinexSVDecoder::CrinexSVDecoder(std::string sv_label, std::size_t n_obs)
    : sv_(std::move(sv_label)),
      n_obs_(n_obs),
      channels_(n_obs),
      filled_(n_obs, 0) {
    for (auto& c : channels_) {
        crinex_channel_reset(&c, 0);
    }
}

std::string CrinexSVDecoder::decode_line(const std::string& line) {
    // Output buffer: 3 (PRN) + n_obs * 16. We'll rstrip at the end.
    std::string out;
    out.reserve(3 + n_obs_ * 16);
    out.append(sv_);
    // Pre-fill with spaces so empty obs render as 16 blanks.
    out.resize(3 + n_obs_ * 16, ' ');

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
                format_f14_3(value, &out[3 + i * 16]);
            }
        }
        // Advance p past the space (if any).
        p = (sp < end) ? sp + 1 : end;
    }

    // Whatever remains is the LLI/SSI delta. Pad to 2*n_obs.
    std::string flag_delta(p, end - p);
    flag_delta.resize(2 * n_obs_, ' ');
    const std::string flag_text = flags_.step(flag_delta);
    // Place flag chars at column 17, 18 of each obs (= positions
    // 3 + i*16 + 14, +15).
    for (std::size_t i = 0; i < n_obs_; ++i) {
        const std::size_t base = 3 + i * 16 + 14;
        out[base]     = (2 * i     < flag_text.size()) ? flag_text[2 * i] : ' ';
        out[base + 1] = (2 * i + 1 < flag_text.size()) ? flag_text[2 * i + 1] : ' ';
    }

    // rstrip.
    std::size_t n = out.size();
    while (n > 0 && out[n - 1] == ' ') --n;
    out.resize(n);
    return out;
}

}  // namespace rinexpy_native
