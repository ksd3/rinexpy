// SPP iterative LSQ kernel implementation.

#include "spp_solve.hpp"

#include <cmath>

namespace rinexpy_native {

namespace {

constexpr double C_LIGHT = 299792458.0;

// 4x4 LU decomposition with partial pivoting + back substitution.
// Returns false if the matrix is singular.
bool solve_4x4(double A[4][4], double b[4], double out[4]) noexcept {
    // Forward elimination with partial pivoting.
    int perm[4] = { 0, 1, 2, 3 };
    for (int k = 0; k < 4; ++k) {
        // Find pivot.
        int piv = k;
        double max_v = std::fabs(A[k][k]);
        for (int i = k + 1; i < 4; ++i) {
            const double v = std::fabs(A[i][k]);
            if (v > max_v) {
                max_v = v;
                piv = i;
            }
        }
        if (max_v < 1e-14) {
            return false;
        }
        if (piv != k) {
            for (int j = 0; j < 4; ++j) {
                std::swap(A[k][j], A[piv][j]);
            }
            std::swap(b[k], b[piv]);
            std::swap(perm[k], perm[piv]);
        }
        // Eliminate.
        for (int i = k + 1; i < 4; ++i) {
            const double m = A[i][k] / A[k][k];
            A[i][k] = 0.0;
            for (int j = k + 1; j < 4; ++j) {
                A[i][j] -= m * A[k][j];
            }
            b[i] -= m * b[k];
        }
    }
    // Back substitution.
    for (int i = 3; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < 4; ++j) {
            s -= A[i][j] * out[j];
        }
        out[i] = s / A[i][i];
    }
    return true;
}

}  // namespace

SppResult spp_solve_iterative(const double* sv_ecef,
                              const double* pseudorange,
                              std::size_t n_sv,
                              const double* init_xyz,
                              double tol, int max_iter,
                              double* out_residuals) noexcept {
    SppResult r;
    r.x = init_xyz[0];
    r.y = init_xyz[1];
    r.z = init_xyz[2];
    r.clock_bias_s = 0.0;

    for (int it = 0; it < max_iter; ++it) {
        // Build normal equations directly: H^T H (4x4) and H^T y (4,)
        // where H[i] = [-LoS_x, -LoS_y, -LoS_z, c] and y[i] = pr[i] -
        // (rho[i] + c*dt). With n_sv small (~10), the cost is
        // dominated by the per-row work.
        double HTH[4][4] = { {0,0,0,0}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} };
        double HTy[4] = { 0, 0, 0, 0 };

        for (std::size_t i = 0; i < n_sv; ++i) {
            const double dx = sv_ecef[3*i+0] - r.x;
            const double dy = sv_ecef[3*i+1] - r.y;
            const double dz = sv_ecef[3*i+2] - r.z;
            const double rho = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (!(rho > 0.0)) continue;
            const double pred = rho + C_LIGHT * r.clock_bias_s;
            const double resid = pseudorange[i] - pred;
            out_residuals[i] = resid;
            const double h0 = -dx / rho;
            const double h1 = -dy / rho;
            const double h2 = -dz / rho;
            const double h3 = C_LIGHT;
            HTH[0][0] += h0 * h0;  HTH[0][1] += h0 * h1;  HTH[0][2] += h0 * h2;  HTH[0][3] += h0 * h3;
            HTH[1][1] += h1 * h1;  HTH[1][2] += h1 * h2;  HTH[1][3] += h1 * h3;
            HTH[2][2] += h2 * h2;  HTH[2][3] += h2 * h3;
            HTH[3][3] += h3 * h3;
            HTy[0] += h0 * resid;
            HTy[1] += h1 * resid;
            HTy[2] += h2 * resid;
            HTy[3] += h3 * resid;
        }
        // Symmetric fill.
        HTH[1][0] = HTH[0][1];
        HTH[2][0] = HTH[0][2]; HTH[2][1] = HTH[1][2];
        HTH[3][0] = HTH[0][3]; HTH[3][1] = HTH[1][3]; HTH[3][2] = HTH[2][3];

        double update[4];
        if (!solve_4x4(HTH, HTy, update)) {
            r.converged = 0;
            r.n_iter = it + 1;
            return r;
        }
        r.x += update[0];
        r.y += update[1];
        r.z += update[2];
        r.clock_bias_s += update[3];

        const double upd_norm = std::sqrt(
            update[0]*update[0] + update[1]*update[1] + update[2]*update[2]);
        r.n_iter = it + 1;
        if (upd_norm < tol) {
            r.converged = 1;
            return r;
        }
    }
    r.converged = 0;
    return r;
}

}  // namespace rinexpy_native
