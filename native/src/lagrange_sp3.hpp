// Order-N Lagrange interpolation for SP3 satellite positions.
//
// SP3 ephemerides are sampled every ~15 minutes; PPP / RTK pipelines
// re-sample them to their own epoch grid (1 Hz, 10 Hz, ...) via a
// per-query Lagrange polynomial through the surrounding source epochs.
// The Python reference in rinexpy.interp does the same thing in pure
// Python; this kernel collapses the per-query weight loop, window
// selection, and tensor contraction into one C++ call.
//
// The arithmetic is intentionally identical to the Python reference so
// values match bit-for-bit on the test corpus.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// Batch interpolate SP3 positions at `n_q` query times.
//
// `src_t`        : (n_src,)            int64 source epoch timestamps
//                                      (ns since epoch, monotonically
//                                      increasing).
// `pos`          : (n_src, n_sv, 3)    float64 row-major positions.
//                                      May contain NaNs; the kernel
//                                      propagates them naturally.
// `query`        : (n_q,)              int64 query timestamps.
// `out`          : (n_q, n_sv, 3)      float64 row-major. Caller-owned.
// `span`         : window size (= order + 1). Clamped to n_src.
//
// For each query the kernel:
//   1. Picks `span` source epochs centred (via binary search) around the
//      query, clamped to the array bounds the same way the Python does.
//   2. Computes Lagrange basis weights at the query time.
//   3. Writes `sum_i w_i * pos[lo+i, sv, c]` into out[q, sv, c].
void interpolate_sp3_lagrange(const std::int64_t* src_t,
                              const double* pos,
                              std::size_t n_src,
                              std::size_t n_sv,
                              const std::int64_t* query,
                              std::size_t n_q,
                              std::size_t span,
                              double* out) noexcept;

}  // namespace rinexpy_native
