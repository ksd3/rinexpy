// Lagrange SP3 interpolation kernel implementation.

#include "lagrange_sp3.hpp"

#include <algorithm>
#include <vector>

namespace rinexpy_native {

namespace {

// Binary-search the largest index `i` with src_t[i] <= q. Returns 0 if
// q < src_t[0]. Matches numpy.searchsorted(side='left') semantics
// closely enough for the SP3 windowing (the +1 offset that the Python
// uses is folded into the "centre on `idx`" arithmetic below).
inline std::size_t searchsorted(const std::int64_t* arr, std::size_t n,
                                std::int64_t q) noexcept {
    std::size_t lo = 0, hi = n;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < q) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

}  // namespace

void interpolate_sp3_lagrange(const std::int64_t* src_t,
                              const double* pos,
                              std::size_t n_src,
                              std::size_t n_sv,
                              const std::int64_t* query,
                              std::size_t n_q,
                              std::size_t span,
                              double* out) noexcept {
    if (span > n_src) span = n_src;
    if (span == 0 || n_src == 0 || n_sv == 0) {
        std::fill(out, out + n_q * n_sv * 3, 0.0);
        return;
    }

    // Per-call scratch: weights vector. Heap-allocated once for the
    // batch — `span` is usually 11 (order-10) but the caller can pick
    // larger if they like.
    std::vector<double> weights(span);
    std::vector<double> sub_t(span);

    const std::size_t row_stride = n_sv * 3;

    for (std::size_t qi = 0; qi < n_q; ++qi) {
        const std::int64_t q = query[qi];

        // Centre the window on the bracketing index, clamp to bounds.
        std::size_t idx = searchsorted(src_t, n_src, q);
        std::size_t lo = (idx > span / 2) ? (idx - span / 2) : 0;
        std::size_t hi = lo + span;
        if (hi > n_src) {
            hi = n_src;
            lo = (hi > span) ? (hi - span) : 0;
        }
        const std::size_t span_actual = hi - lo;

        // Source-time slice as float64 (the Python does the same cast).
        for (std::size_t i = 0; i < span_actual; ++i) {
            sub_t[i] = static_cast<double>(src_t[lo + i]);
        }
        const double x = static_cast<double>(q);

        // Compute basis weights w_i = prod_{j != i} (x - t_j) / (t_i - t_j).
        for (std::size_t i = 0; i < span_actual; ++i) {
            double w = 1.0;
            const double t_i = sub_t[i];
            for (std::size_t j = 0; j < span_actual; ++j) {
                if (j == i) continue;
                w *= (x - sub_t[j]) / (t_i - sub_t[j]);
            }
            weights[i] = w;
        }

        // tensordot(weights, pos[lo:hi], axes=(0, 0)) -> (n_sv, 3).
        // Tight inner: stride through SVs and components contiguously.
        double* out_row = out + qi * row_stride;
        // Initialise this row to zero so NaN-free entries don't pick up
        // garbage and so NaN inputs propagate (NaN * 0 = NaN, NaN + 0 =
        // NaN, but NaN + finite = NaN, so any node NaN poisons its
        // (sv, component) output -- matches the Python reference).
        std::fill(out_row, out_row + row_stride, 0.0);
        for (std::size_t i = 0; i < span_actual; ++i) {
            const double w = weights[i];
            const double* src_row = pos + (lo + i) * row_stride;
            for (std::size_t k = 0; k < row_stride; ++k) {
                out_row[k] += w * src_row[k];
            }
        }
    }
}

}  // namespace rinexpy_native
