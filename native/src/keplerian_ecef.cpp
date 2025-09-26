// Keplerian -> ECEF kernel implementation.

#include "keplerian_ecef.hpp"

#include <cmath>

namespace rinexpy_native {

void keplerian_to_ecef_batch(
    const double* M0, const double* dn, const double* e, const double* sqrtA,
    const double* omega, const double* Cuc, const double* Cus,
    const double* Cic, const double* Cis, const double* Crc, const double* Crs,
    const double* Io, const double* IDOT, const double* Omega0,
    const double* OmegaDot, const double* Toe, const double* tk,
    std::size_t n, double* out_xyz) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const double e_i = e[i];
        const double A = sqrtA[i] * sqrtA[i];
        const double n0 = std::sqrt(GM_EARTH / (A * A * A));
        const double n_mean = n0 + dn[i];

        const double tki = tk[i];
        const double Mk = M0[i] + n_mean * tki;

        // Newton iteration for Kepler's eq with early exit.
        double Ek = Mk;
        for (int it = 0; it < 8; ++it) {
            const double sinE = std::sin(Ek);
            const double cosE = std::cos(Ek);
            const double f = Ek - e_i * sinE - Mk;
            const double fp = 1.0 - e_i * cosE;
            const double delta = f / fp;
            Ek -= delta;
            if (std::fabs(delta) < 1e-14) break;
        }
        const double sinE = std::sin(Ek);
        const double cosE = std::cos(Ek);
        const double nu = std::atan2(std::sqrt(1.0 - e_i * e_i) * sinE,
                                     cosE - e_i);

        const double phi = nu + omega[i];
        const double cos2p = std::cos(2.0 * phi);
        const double sin2p = std::sin(2.0 * phi);
        const double duk = Cuc[i] * cos2p + Cus[i] * sin2p;
        const double dik = Cic[i] * cos2p + Cis[i] * sin2p;
        const double drk = Crc[i] * cos2p + Crs[i] * sin2p;

        const double uk = phi + duk;
        const double ik = Io[i] + IDOT[i] * tki + dik;
        const double rk = A * (1.0 - e_i * cosE) + drk;

        const double Omega = Omega0[i]
                           + (OmegaDot[i] - OMEGA_E_RAD_S) * tki
                           - OMEGA_E_RAD_S * Toe[i];

        const double cos_u = std::cos(uk);
        const double sin_u = std::sin(uk);
        const double cos_O = std::cos(Omega);
        const double sin_O = std::sin(Omega);
        const double cos_i = std::cos(ik);
        const double sin_i = std::sin(ik);

        const double Xk1 = rk * cos_u;
        const double Yk1 = rk * sin_u;
        out_xyz[3 * i + 0] = Xk1 * cos_O - Yk1 * sin_O * cos_i;
        out_xyz[3 * i + 1] = Xk1 * sin_O + Yk1 * cos_O * cos_i;
        out_xyz[3 * i + 2] = Yk1 * sin_i;
    }
}

}  // namespace rinexpy_native
