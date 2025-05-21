// LAMBDA ILS kernel implementation.

#include "lambda_ils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace rinexpy_native {

namespace {

// Index helpers for row-major n*n matrices.
inline std::size_t IDX(std::size_t i, std::size_t j, std::size_t n) {
    return i * n + j;
}

}  // namespace

void ldl(const double* Q, std::size_t n,
         double* L_out, double* D_out) {
    // Mirror the Python algorithm: top-down LDL with unit-diagonal L,
    // diagonal D. Identity init for L, zero D.
    for (std::size_t i = 0; i < n * n; ++i) {
        L_out[i] = 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        L_out[IDX(i, i, n)] = 1.0;
        D_out[i] = 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        double diag = Q[IDX(i, i, n)];
        for (std::size_t k = 0; k < i; ++k) {
            const double Lk = L_out[IDX(i, k, n)];
            diag -= Lk * Lk * D_out[k];
        }
        if (diag <= 0.0) {
            throw LdlNotPositiveDefinite("Q is not positive definite");
        }
        D_out[i] = diag;
        for (std::size_t j = i + 1; j < n; ++j) {
            double s = Q[IDX(j, i, n)];
            for (std::size_t k = 0; k < i; ++k) {
                s -= L_out[IDX(j, k, n)] * L_out[IDX(i, k, n)] * D_out[k];
            }
            L_out[IDX(j, i, n)] = s / diag;
        }
    }
}

void bootstrap(const double* L, const double* a_float, std::size_t n,
               std::int64_t* a_int_out) {
    // Conditional rounding from index n-1 down to 0.
    std::vector<double> a_cond(a_float, a_float + n);
    for (std::size_t ip1 = n; ip1 > 0; --ip1) {
        const std::size_t i = ip1 - 1;
        const std::int64_t rounded = static_cast<std::int64_t>(
            std::llround(a_cond[i]));
        a_int_out[i] = rounded;
        const double diff = a_cond[i] - static_cast<double>(rounded);
        for (std::size_t j = 0; j < i; ++j) {
            a_cond[j] -= L[IDX(i, j, n)] * diff;
        }
    }
}

namespace {

// Hash for int-vector dedup. Matches the Python `tuple(int(v) for v
// in current)` set membership lookup. Vector length is fixed across
// the search so we use a simple FNV-1a mix.
struct VecHash {
    std::size_t operator()(const std::vector<std::int64_t>& v) const noexcept {
        std::size_t h = 1469598103934665603ULL;
        for (auto x : v) {
            h ^= static_cast<std::size_t>(x);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// Per-search mutable state. Keeps the pruning bound, node budget,
// and the running list of candidates sorted ascending by sq error.
struct SearchState {
    std::size_t n;
    std::size_t n_cands;
    std::uint64_t max_nodes;
    double max_seconds;
    const double* a_float;
    const double* L;
    const double* D;
    std::chrono::steady_clock::time_point t_start;

    double bound = std::numeric_limits<double>::infinity();
    std::uint64_t nodes = 0;
    int aborted_reason = 0;

    // (sq_err, candidate). Kept sorted ascending by sq_err. Capped at
    // 4*n_cands during the search to keep ordering noise from
    // dropping a real candidate.
    std::vector<std::pair<double, std::vector<std::int64_t>>> cands;
    std::unordered_set<std::vector<std::int64_t>, VecHash> seen;
};

// Conditional mean of a_float[idx] given the partial integer vector
// in `current` for indices > idx.
inline double conditional(const SearchState& st, std::size_t idx,
                          const std::vector<std::int64_t>& current) {
    double c = st.a_float[idx];
    for (std::size_t j = idx + 1; j < st.n; ++j) {
        c -= st.L[IDX(j, idx, st.n)] *
             (static_cast<double>(current[j]) - st.a_float[j]);
    }
    return c;
}

// Returns false to unwind on abort.
bool search(SearchState& st, std::ptrdiff_t idx,
            std::vector<std::int64_t>& current, double residual_sq) {
    if (st.aborted_reason != 0) return false;
    ++st.nodes;
    if (st.nodes > st.max_nodes) {
        st.aborted_reason = 1;
        return false;
    }
    if (st.max_seconds > 0.0 && (st.nodes & 0xFFF) == 0) {
        // Sample the clock infrequently — chrono::now is ~30 ns and
        // dominates the inner loop otherwise.
        const auto dt = std::chrono::steady_clock::now() - st.t_start;
        const double secs = std::chrono::duration<double>(dt).count();
        if (secs > st.max_seconds) {
            st.aborted_reason = 2;
            return false;
        }
    }
    if (residual_sq >= st.bound) return true;

    if (idx < 0) {
        // Complete candidate.
        if (st.seen.find(current) != st.seen.end()) return true;
        st.seen.insert(current);
        st.cands.emplace_back(residual_sq, current);
        std::sort(st.cands.begin(), st.cands.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });
        const std::size_t cap = st.n_cands * 4;
        if (st.cands.size() > cap) {
            for (std::size_t k = cap; k < st.cands.size(); ++k) {
                st.seen.erase(st.cands[k].second);
            }
            st.cands.resize(cap);
        }
        if (st.cands.size() >= st.n_cands) {
            st.bound = st.cands[st.n_cands - 1].first;
        }
        return true;
    }

    const double c = conditional(st, static_cast<std::size_t>(idx), current);
    const std::int64_t center = static_cast<std::int64_t>(std::llround(c));
    const double D_idx = st.D[idx];

    // Visit integers in increasing |delta|: 0, +1, -1, +2, -2, ...
    // Break the outer loop once neither sign at level k stays in the
    // pruning bound. Maximum |delta| of 40 mirrors the Python search.
    for (int k = 0; k < 40; ++k) {
        bool any_in_bound = false;
        const int signs_n = (k == 0) ? 1 : 2;
        const int sign_arr[2] = { 1, -1 };
        for (int si = 0; si < signs_n; ++si) {
            const std::int64_t cand = center + sign_arr[si] * k;
            const double delta = static_cast<double>(cand) - c;
            const double contrib = (delta * delta) / D_idx;
            const double new_sq = residual_sq + contrib;
            if (new_sq >= st.bound) continue;
            any_in_bound = true;
            current[idx] = cand;
            if (!search(st, idx - 1, current, new_sq)) return false;
        }
        if (!any_in_bound && k > 0) break;
    }
    return true;
}

// Compute (a_int - a_float)^T Q^{-1} (a_int - a_float) using the LDL
// factor. Mirrors Python's _squared_residual.
double squared_residual(const std::int64_t* a_int, const double* a_float,
                        const double* L, const double* D, std::size_t n) {
    std::vector<double> diff(n);
    for (std::size_t i = 0; i < n; ++i) {
        diff[i] = a_float[i];
    }
    double res = 0.0;
    for (std::size_t ip1 = n; ip1 > 0; --ip1) {
        const std::size_t i = ip1 - 1;
        const double d = static_cast<double>(a_int[i]) - diff[i];
        res += (d * d) / D[i];
        for (std::size_t j = 0; j < i; ++j) {
            diff[j] -= L[IDX(i, j, n)] *
                       (static_cast<double>(a_int[i]) - diff[i]);
        }
    }
    return res;
}

}  // namespace

IlsResult integer_least_squares(const double* a_float,
                                const double* Q,
                                std::size_t n,
                                std::size_t n_cands,
                                std::uint64_t max_nodes,
                                double max_seconds) {
    if (n_cands == 0) n_cands = 1;

    std::vector<double> L(n * n);
    std::vector<double> D(n);
    ldl(Q, n, L.data(), D.data());

    std::vector<std::int64_t> boot(n);
    bootstrap(L.data(), a_float, n, boot.data());

    SearchState st;
    st.n = n;
    st.n_cands = n_cands;
    st.max_nodes = max_nodes;
    st.max_seconds = max_seconds;
    st.a_float = a_float;
    st.L = L.data();
    st.D = D.data();
    st.t_start = std::chrono::steady_clock::now();

    std::vector<std::int64_t> current(boot);
    search(st, static_cast<std::ptrdiff_t>(n) - 1, current, 0.0);

    IlsResult out;
    if (st.cands.empty()) {
        // Budget cut us off before any complete path landed. Fall back
        // to the bootstrap as the partial.
        const double boot_sq = squared_residual(
            boot.data(), a_float, L.data(), D.data(), n);
        st.cands.emplace_back(boot_sq, std::move(boot));
    }
    std::sort(st.cands.begin(), st.cands.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    if (st.cands.size() > n_cands) st.cands.resize(n_cands);

    const std::size_t k = st.cands.size();
    out.n_returned = k;
    out.candidates.resize(k * n);
    out.sq_errors.resize(k);
    for (std::size_t i = 0; i < k; ++i) {
        out.sq_errors[i] = st.cands[i].first;
        for (std::size_t j = 0; j < n; ++j) {
            out.candidates[i * n + j] = st.cands[i].second[j];
        }
    }
    out.L_factor = std::move(L);
    out.nodes_visited = st.nodes;
    out.aborted_reason = st.aborted_reason;
    return out;
}

}  // namespace rinexpy_native
