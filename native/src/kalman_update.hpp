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

// One PPP filter scalar update.
//
// `x`           (n,)          state vector (will be updated in place)
// `P`           (n, n) row-major covariance (updated in place)
// `n`                          state dimension
// `u`           (3,)            unit LoS vector from rx to SV (= -diff/rho)
// `is_phase`                    false for code obs; true for phase
// `sv_index`                    SV slot index when is_phase=true (-1 otherwise)
// `obs`                          measurement value (already corrected for
//                                sat clock + tropo etc.)
// `rho`                          predicted geometric range
// `r`                            measurement variance (sigma^2)
//
// State layout assumed: x[0..3] = position + clock,
// x[4 + sv_index] = iono-free ambiguity (when is_phase).
void kalman_scalar_update_static_ppp(
    double* x, double* P, std::size_t n,
    const double* u, bool is_phase, int sv_index,
    double obs, double rho, double r) noexcept;

}  // namespace rinexpy_native
