// Single-point positioning (SPP) iterative LSQ kernel.
//
// The Python reference in rinexpy.positioning.spp_solve does ~5
// iterations of:
//
//   1. compute the (n_sv,) geometric range vector,
//   2. build the (n_sv, 4) design matrix,
//   3. call np.linalg.lstsq for the 4-parameter update,
//   4. update the state.
//
// Each iteration crosses the Python/numpy boundary several times.
// The kernel collapses all iterations into one C++ call and uses a
// hand-rolled 4x4 normal-equations solve (the geometry is small
// enough that the constant factor of dense numpy linalg is wasted).

#pragma once

#include <cstddef>

namespace rinexpy_native {

// Result of one SPP solve.
struct SppResult {
    double x = 0.0, y = 0.0, z = 0.0;  // ECEF position (m)
    double clock_bias_s = 0.0;
    int n_iter = 0;
    int converged = 0;                  // 1 if |update[:3]| < tol
    // Residuals at the converged iteration; caller must pre-allocate
    // n_sv slots in `out_residuals`.
};

// Solve for `(position, clock_bias)` from `n_sv` pseudoranges.
//
// `sv_ecef`       (n_sv, 3) float64 row-major SV ECEF positions
// `pseudorange`   (n_sv,) float64 measured ranges
// `init_xyz`      (3,) initial guess
// `tol`           convergence tolerance on |update[:3]| in meters
// `max_iter`      iteration cap
// `out_residuals` (n_sv,) float64 caller-allocated; filled with final residuals
SppResult spp_solve_iterative(const double* sv_ecef,
                              const double* pseudorange,
                              std::size_t n_sv,
                              const double* init_xyz,
                              double tol, int max_iter,
                              double* out_residuals) noexcept;

}  // namespace rinexpy_native
