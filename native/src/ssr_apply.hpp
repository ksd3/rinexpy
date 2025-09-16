// SSR (State-Space Representation) orbit + clock correction application.
//
// The Python references in rinexpy.realtime.RealtimeOrbitClock.{
// apply_orbit_correction, apply_clock_correction} do per-call numpy
// allocations for the radial/along/cross frame unit vectors plus a
// handful of cross-product / norm calls. The C++ kernel collapses
// those into one stack-allocated 3-vector pass with no allocation.
//
// The cache itself stays in Python — only the per-correction math
// moves here. That's enough to lift a 10 Hz multi-GNSS realtime-PPP
// loop off the SSR-application hot spot.

#pragma once

namespace rinexpy_native {

// Apply an SSR orbit correction (radial / along / cross) plus its
// linear rate to a broadcast ECEF position.
//
// `r_in`         (3,) broadcast ECEF position (m)
// `v_in`         (3,) broadcast ECEF velocity (m/s) — required to
//                build the RAC frame
// `rac0`         (3,) orbit correction at the SSR epoch
//                (radial_m, along_m, cross_m)
// `racdot`       (3,) orbit correction rate (m/s)
// `elapsed_s`    seconds since the correction was received
// `r_out`        (3,) corrected ECEF position written here
void apply_ssr_orbit_correction(const double* r_in, const double* v_in,
                                const double* rac0, const double* racdot,
                                double elapsed_s, double* r_out) noexcept;

// Apply an SSR clock correction to a broadcast clock bias in seconds.
//
// `broadcast_clock_s`   broadcast clock bias (s)
// `c0`, `c1`, `c2`      SSR clock polynomial coefficients (m, m/s, m/s^2)
// `elapsed_s`           seconds since correction received
//
// IGS convention: precise = broadcast - delta / c.
double apply_ssr_clock_correction(double broadcast_clock_s,
                                  double c0, double c1, double c2,
                                  double elapsed_s) noexcept;

}  // namespace rinexpy_native
