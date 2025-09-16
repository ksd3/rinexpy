// SSR correction-application kernel implementation.

#include "ssr_apply.hpp"

#include <cmath>

namespace rinexpy_native {

namespace {

constexpr double C_LIGHT = 299792458.0;

inline double norm3(const double* v) noexcept {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

}  // namespace

void apply_ssr_orbit_correction(const double* r_in, const double* v_in,
                                const double* rac0, const double* racdot,
                                double elapsed_s, double* r_out) noexcept {
    // RAC frame unit vectors (e_r, e_a, e_c).
    const double rn = norm3(r_in);
    if (!(rn > 0.0)) {
        // Pathological input: leave the position unchanged.
        r_out[0] = r_in[0]; r_out[1] = r_in[1]; r_out[2] = r_in[2];
        return;
    }
    const double e_r[3] = { r_in[0]/rn, r_in[1]/rn, r_in[2]/rn };

    // h = r x v
    const double hx = r_in[1]*v_in[2] - r_in[2]*v_in[1];
    const double hy = r_in[2]*v_in[0] - r_in[0]*v_in[2];
    const double hz = r_in[0]*v_in[1] - r_in[1]*v_in[0];
    const double hn = std::sqrt(hx*hx + hy*hy + hz*hz);
    if (!(hn > 0.0)) {
        r_out[0] = r_in[0]; r_out[1] = r_in[1]; r_out[2] = r_in[2];
        return;
    }
    const double e_c[3] = { hx/hn, hy/hn, hz/hn };

    // e_a = e_c x e_r
    const double e_a[3] = {
        e_c[1]*e_r[2] - e_c[2]*e_r[1],
        e_c[2]*e_r[0] - e_c[0]*e_r[2],
        e_c[0]*e_r[1] - e_c[1]*e_r[0],
    };

    const double d_r = rac0[0] + racdot[0] * elapsed_s;
    const double d_a = rac0[1] + racdot[1] * elapsed_s;
    const double d_c = rac0[2] + racdot[2] * elapsed_s;

    r_out[0] = r_in[0] - (d_r*e_r[0] + d_a*e_a[0] + d_c*e_c[0]);
    r_out[1] = r_in[1] - (d_r*e_r[1] + d_a*e_a[1] + d_c*e_c[1]);
    r_out[2] = r_in[2] - (d_r*e_r[2] + d_a*e_a[2] + d_c*e_c[2]);
}

double apply_ssr_clock_correction(double broadcast_clock_s,
                                  double c0, double c1, double c2,
                                  double elapsed_s) noexcept {
    const double delta_m = c0 + c1 * elapsed_s
                         + 0.5 * c2 * elapsed_s * elapsed_s;
    return broadcast_clock_s - delta_m / C_LIGHT;
}

}  // namespace rinexpy_native
