// LAMBDA integer least-squares kernel (LDL + bootstrap + branch-and-bound).
//
// All routines operate on row-major dense matrices of doubles. The
// public interface returns a small result struct that the nanobind
// layer pivots into NumPy arrays for the Python caller; the search
// itself never touches the GIL.
//
// Numerical contract matches rinexpy.lambda_ar.{ldl, bootstrap,
// integer_least_squares} bit-for-bit on the bundled tests.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rinexpy_native {

// Exception thrown when the LDL step finds a non-positive diagonal,
// matching numpy.linalg.LinAlgError on the Python side.
struct LdlNotPositiveDefinite : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Compute Q = L D L^T with L unit-lower-triangular and D > 0.
// `Q`, `L_out` are row-major (n*n); `D_out` is length n. Q is read
// only — pass a defensive copy if the caller needs to keep it.
void ldl(const double* Q, std::size_t n,
         double* L_out, double* D_out);

// Bootstrap integer estimate. `L` is the LDL factor (unit lower
// triangular, row-major). `a_float` is the n-vector input. Writes an
// n-vector of int64 (matches numpy default int on 64-bit Linux/macOS)
// into `a_int_out`.
void bootstrap(const double* L, const double* a_float, std::size_t n,
               std::int64_t* a_int_out);

// Result of an integer-LS search: up to `n_cands` integer candidates
// and their associated squared residuals, sorted ascending.
struct IlsResult {
    // candidates: row-major (n_returned, n) ints.
    std::vector<std::int64_t> candidates;
    // sq_errors: length n_returned.
    std::vector<double> sq_errors;
    // n_returned <= requested n_cands.
    std::size_t n_returned = 0;
    // Search budget bookkeeping.
    std::uint64_t nodes_visited = 0;
    // 0 = ran to completion. Any non-zero value mirrors the Python
    // ILSAborted reason: 1 = max_nodes exceeded, 2 = max_seconds.
    int aborted_reason = 0;
};

// Branch-and-bound ILS search.
//
// `a_float`, `Q` are length-n and n*n row-major inputs. The search
// keeps up to `n_cands` (>=1) best candidates by squared residual.
//
// `max_nodes` caps the total node visits across the search tree; on
// overflow, the function returns with `aborted_reason = 1` and the
// best partial candidates available. If `max_seconds > 0`, the search
// also aborts when wall-clock budget is exceeded (aborted_reason=2);
// pass <= 0 for no wall-clock limit.
IlsResult integer_least_squares(const double* a_float,
                                const double* Q,
                                std::size_t n,
                                std::size_t n_cands,
                                std::uint64_t max_nodes,
                                double max_seconds);

}  // namespace rinexpy_native
