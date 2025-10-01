// GPT2w cell evaluator implementation.

#include "gpt2w_eval.hpp"

#include <cmath>

namespace rinexpy_native {

namespace {

constexpr double TWO_PI = 6.283185307179586;
constexpr double FOUR_PI = 12.566370614359172;
constexpr double YEAR_DAYS = 365.25;

// Per-quantity seasonal evaluation. coefs is the 5-element layout
// (mean, a1, b1, a2, b2). cos_a / sin_a are the annual phase trig
// terms; cos_b / sin_b are the semi-annual.
inline double seasonal(const double* coefs, double cos_a, double sin_a,
                       double cos_b, double sin_b) noexcept {
    return coefs[0] + coefs[1] * cos_a + coefs[2] * sin_a
                    + coefs[3] * cos_b + coefs[4] * sin_b;
}

// Evaluate the 8 quantities at one of the 4 corner cells. Writes
// undu, ref_h, p, t, q, dt, tm, lam, a_h, a_w into out[0..9].
inline void eval_corner(const double* cell,
                        double cos_a, double sin_a,
                        double cos_b, double sin_b,
                        double* out) noexcept {
    out[0] = cell[0];          // undu
    out[1] = cell[1];          // ref_h
    out[2] = seasonal(cell + 2,  cos_a, sin_a, cos_b, sin_b);  // p
    out[3] = seasonal(cell + 7,  cos_a, sin_a, cos_b, sin_b);  // t
    out[4] = seasonal(cell + 12, cos_a, sin_a, cos_b, sin_b);  // q
    out[5] = seasonal(cell + 17, cos_a, sin_a, cos_b, sin_b);  // dt
    out[6] = seasonal(cell + 22, cos_a, sin_a, cos_b, sin_b);  // tm
    out[7] = seasonal(cell + 27, cos_a, sin_a, cos_b, sin_b);  // lam
    out[8] = seasonal(cell + 32, cos_a, sin_a, cos_b, sin_b);  // a_h
    out[9] = seasonal(cell + 37, cos_a, sin_a, cos_b, sin_b);  // a_w
}

// Bilinear interpolation of one column across the 4 corner buffers.
inline double bilerp(double v00, double v01, double v10, double v11,
                     double w_lat, double w_lon) noexcept {
    const double v0 = v00 * (1.0 - w_lon) + v01 * w_lon;
    const double v1 = v10 * (1.0 - w_lon) + v11 * w_lon;
    return v0 * (1.0 - w_lat) + v1 * w_lat;
}

}  // namespace

void gpt2w_eval_cell(const double* cells, double w_lat, double w_lon,
                     double doy, double altitude_m,
                     double* out) noexcept {
    // Compute the 4 seasonal trig terms once.
    const double phase = (doy - 1.0) / YEAR_DAYS;
    const double cos_a = std::cos(TWO_PI * phase);
    const double sin_a = std::sin(TWO_PI * phase);
    const double cos_b = std::cos(FOUR_PI * phase);
    const double sin_b = std::sin(FOUR_PI * phase);

    double corner[4][10];
    eval_corner(cells +   0,  cos_a, sin_a, cos_b, sin_b, corner[0]);
    eval_corner(cells +  44,  cos_a, sin_a, cos_b, sin_b, corner[1]);
    eval_corner(cells +  88,  cos_a, sin_a, cos_b, sin_b, corner[2]);
    eval_corner(cells + 132,  cos_a, sin_a, cos_b, sin_b, corner[3]);

    auto B = [&](int col) {
        return bilerp(corner[0][col], corner[1][col],
                      corner[2][col], corner[3][col], w_lat, w_lon);
    };
    const double p0   = B(2);
    const double t0_  = B(3);
    const double q    = B(4);
    const double dt   = B(5);
    /* tm  unused for return */
    const double lam  = B(7);
    const double a_h  = B(8);
    const double a_w  = B(9);
    const double undu = B(0);
    const double ref_h= B(1);

    const double dh = altitude_m - ref_h;
    const double t_h = t0_ + dt * dh;
    const double e0 = q * p0 / (0.622 + 0.378 * q);
    const double p_h = p0 * std::pow(1.0 - 0.0000226 * dh, 5.225);
    const double e_h = e0 * std::pow(p_h / p0, lam + 1.0);

    out[0] = p_h;           // pressure_hpa
    out[1] = t_h;           // temperature_k
    out[2] = e_h;           // e_hpa
    out[3] = a_h;
    out[4] = a_w;
    out[5] = dt;            // T_lapse
    out[6] = undu;
}

}  // namespace rinexpy_native
