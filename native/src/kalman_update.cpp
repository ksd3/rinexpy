// EKF scalar-update kernel implementation.
//
// Derivation. With H a row-vector (size n) and r the scalar variance:
//
//   K  = P H^T / s         where s = H P H^T + r
//   x' = x + K * innov
//   P' = (I - K H) P (I - K H)^T + K K^T r
//
// Algebraically (writing h = H^T as a column vector):
//
//   A = (I - K H) P = P - K (H P) = P - k * hp     where hp = H P
//   P' = A (I - K H)^T = A - A H^T K^T = A - (A h) k^T = A - q k^T
//        + K K^T r
//
// where q = A h. With H having only ~4 nonzeros (indices 0..3 always,
// plus 4+sv_index when is_phase), computing hp and q is O(n) instead
// of O(n^2). The remaining O(n^2) work is the outer-product updates,
// which we do with two dense loops.

#include "kalman_update.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace rinexpy_native {

namespace {

// Inline helper to index row-major (i, j) of an n x n matrix.
inline std::size_t IJ(std::size_t i, std::size_t j, std::size_t n) noexcept {
    return i * n + j;
}

}  // namespace

void kalman_scalar_update_static_ppp(
    double* x, double* P, std::size_t n,
    const double* u, bool is_phase, int sv_index,
    double obs, double rho, double r) noexcept {
    // Predicted measurement.
    double pred = rho + x[3];
    if (is_phase && sv_index >= 0) {
        pred += x[4 + sv_index];
    }
    const double innov = obs - pred;

    // Sparse H: indices and values.
    // Indices [0, 1, 2, 3] = u[0..2] + 1.0; optional [4 + sv_index] = 1.0.
    int h_idx[5];
    double h_val[5];
    int hn = 4;
    h_idx[0] = 0; h_val[0] = u[0];
    h_idx[1] = 1; h_val[1] = u[1];
    h_idx[2] = 2; h_val[2] = u[2];
    h_idx[3] = 3; h_val[3] = 1.0;
    if (is_phase && sv_index >= 0) {
        h_idx[4] = 4 + sv_index;
        h_val[4] = 1.0;
        hn = 5;
    }

    // hp = H @ P; column-major access of P (P is symmetric so row=col).
    std::vector<double> hp(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double s_ = 0.0;
        for (int a = 0; a < hn; ++a) {
            s_ += h_val[a] * P[IJ(static_cast<std::size_t>(h_idx[a]), j, n)];
        }
        hp[j] = s_;
    }

    // s = H P H^T + r = sum_a h_val[a] * hp[h_idx[a]] + r.
    double s_scalar = r;
    for (int a = 0; a < hn; ++a) {
        s_scalar += h_val[a] * hp[static_cast<std::size_t>(h_idx[a])];
    }
    if (!(s_scalar > 0.0)) {
        // Mirror the Python: refuse to invert nonpositive innovation cov.
        return;
    }

    // K = P H^T / s.  P H^T row i = sum_a h_val[a] P[i, h_idx[a]]. With
    // P symmetric this is hp by columns but using rows of P (same
    // values). For sparse H, compute K directly column-by-column.
    std::vector<double> k(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s_ = 0.0;
        for (int a = 0; a < hn; ++a) {
            s_ += h_val[a] * P[IJ(i, static_cast<std::size_t>(h_idx[a]), n)];
        }
        k[i] = s_ / s_scalar;
    }

    // State update.
    for (std::size_t i = 0; i < n; ++i) {
        x[i] += k[i] * innov;
    }

    // A = P - outer(k, hp). Row-major: A[i,j] = P[i,j] - k[i] * hp[j].
    // Compute and store A in P directly; that's the standard
    // covariance update except we still need q = A @ h before doing
    // the final correction. We compute q on the fly while writing A.

    // First overwrite P with A (P_new = P - k * hp^T).
    for (std::size_t i = 0; i < n; ++i) {
        const double ki = k[i];
        for (std::size_t j = 0; j < n; ++j) {
            P[IJ(i, j, n)] -= ki * hp[j];
        }
    }

    // q = A h: q[i] = sum_a h_val[a] * A[i, h_idx[a]].
    std::vector<double> q(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s_ = 0.0;
        for (int a = 0; a < hn; ++a) {
            s_ += h_val[a] * P[IJ(i, static_cast<std::size_t>(h_idx[a]), n)];
        }
        q[i] = s_;
    }

    // Final: P' = A - outer(q, k) + outer(k, k) * r.
    for (std::size_t i = 0; i < n; ++i) {
        const double qi = q[i];
        const double ki = k[i];
        const double ki_r = ki * r;
        for (std::size_t j = 0; j < n; ++j) {
            P[IJ(i, j, n)] += -qi * k[j] + ki_r * k[j];
        }
    }
}

}  // namespace rinexpy_native
