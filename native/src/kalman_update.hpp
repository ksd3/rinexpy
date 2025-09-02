// Scalar EKF measurement-update kernel for the rinexpy PPP filters.
//
// The Python references in rinexpy.kalman, rinexpy.kalman_multignss,
// and rinexpy.kalman_ztd all share the same inner step: per
// observation, build a sparse H row, compute the Kalman gain K from
// the current covariance P, and apply a Joseph-form covariance
// update. State dimensions are tiny (4 + n_sv ~= 40) but the call
// frequency is huge (millions of updates over a 24 h replay).
//
// The Python implementation pays O(n^3) per update because the
// I_KH @ P @ I_KH.T contraction goes through full numpy matmul.
// This kernel exploits the sparse structure of H (4 or 5 nonzeros
// out of n) to drop the cost to O(n^2). The arithmetic is
// algebraically identical to the Joseph form so values stay
// bit-stable against the Python reference.

#pragma once

#include <cstddef>

namespace rinexpy_native {

// Generic Joseph-form scalar EKF update with sparse H.
//
// `x`         (n,)            state (updated in place)
// `P`         (n, n)          row-major covariance (updated in place)
// `n`                          state dimension
// `h_indices` (hn,)            column indices of H's nonzero entries
// `h_values`  (hn,)            H values at those columns
// `hn`                          number of nonzeros (1..n)
// `innovation`                  obs - pred (caller computes this since
//                              the predicted-measurement formula is
//                              filter-specific)
// `r`                          measurement variance
//
// O(n^2) instead of the O(n^3) the dense numpy version pays.
void kalman_scalar_update_sparse(
    double* x, double* P, std::size_t n,
    const int* h_indices, const double* h_values, int hn,
    double innovation, double r) noexcept;

}  // namespace rinexpy_native
