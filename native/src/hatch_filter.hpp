// Carrier-smoothed code (Hatch filter) kernel.
//
// The Python reference in rinexpy.qc.hatch_filter walks each epoch
// with a Python-level for-loop that does isfinite checks, slip
// branching, and the recursion P_s[k] = (P[k] + (m-1)*(P_s[k-1] +
// (phi[k]-phi[k-1]))) / m. The math is trivial but the per-epoch
// Python overhead dominates; a C++ scalar loop handles the same
// recursion at register speed.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// Hatch-filter one per-SV time series.
//
// `pr`       (n,)  raw code pseudorange in meters (NaN for missing)
// `phi`      (n,)  carrier phase in meters (NaN for missing)
// `slips`    (n,)  optional uint8 mask (1 at slip epochs, 0 otherwise).
//                  Pass nullptr to disable.
// `n`               length of all three series
// `window`           max smoothing window
// `out`              (n,) caller-allocated output buffer (float64).
void hatch_filter_kernel(const double* pr, const double* phi,
                         const std::uint8_t* slips,
                         std::size_t n, int window,
                         double* out) noexcept;

}  // namespace rinexpy_native
