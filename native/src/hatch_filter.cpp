// Hatch filter kernel implementation.

#include "hatch_filter.hpp"

#include <cmath>
#include <limits>

namespace rinexpy_native {

void hatch_filter_kernel(const double* pr, const double* phi,
                         const std::uint8_t* slips,
                         std::size_t n, int window,
                         double* out) noexcept {
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    int m = 0;
    double prev_phi = nan_v;
    double prev_out = nan_v;

    for (std::size_t k = 0; k < n; ++k) {
        const double pr_k = pr[k];
        const double phi_k = phi[k];
        if (!(std::isfinite(pr_k) && std::isfinite(phi_k))) {
            out[k] = nan_v;
            m = 0;
            prev_phi = nan_v;
            continue;
        }
        const bool slip_here = (slips != nullptr) && (slips[k] != 0);
        if (m == 0 || slip_here || !std::isfinite(prev_phi)) {
            out[k] = pr_k;
            m = 1;
        } else {
            if (m + 1 <= window) m += 1;
            // Match the Python: P_s[k] = (P[k] + (m-1)*(P_s[k-1] + (phi[k]-prev_phi))) / m
            out[k] = (pr_k + (m - 1) * (prev_out + (phi_k - prev_phi))) / m;
        }
        prev_phi = phi_k;
        prev_out = out[k];
    }
}

}  // namespace rinexpy_native
