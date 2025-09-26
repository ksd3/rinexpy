// Batched Keplerian -> ECEF kernel for GPS / Galileo navigation
// ephemerides.
//
// Mirrors the math in rinexpy.keplerian.keplerian2ecef but moves the
// Newton iteration into a per-SV loop with early-exit on convergence.
// The numpy version does all 8 iterations on every SV regardless of
// whether it converged; for e ~ 0.01 (typical GPS), the kernel
// usually converges in 2-3 iterations.

#pragma once

#include <cstddef>

namespace rinexpy_native {

// Earth constants matching keplerian.py.
inline constexpr double GM_EARTH = 3.986004418e14;
inline constexpr double OMEGA_E_RAD_S = 7.2921151467e-5;

// Compute ECEF positions for `n` satellites with parallel arrays of
// Keplerian fields. `out_xyz` is row-major (n, 3) float64.
//
// All arrays are length n. `tk` is the time-of-applicability offset
// in seconds (computed by the Python wrapper using the same GPS-epoch
// arithmetic as the reference).
void keplerian_to_ecef_batch(
    const double* M0, const double* dn, const double* e, const double* sqrtA,
    const double* omega, const double* Cuc, const double* Cus,
    const double* Cic, const double* Cis, const double* Crc, const double* Crs,
    const double* Io, const double* IDOT, const double* Omega0,
    const double* OmegaDot, const double* Toe, const double* tk,
    std::size_t n, double* out_xyz) noexcept;

}  // namespace rinexpy_native
