// CRINEX k-th order integer-differencing kernel implementation.
//
// The Hatanaka 2008 paper describes the scheme in section 3 (eqns
// 1-3). For order k:
//
//   For epoch 0 .. k-1, transmit the absolute integer value Y_n
//   directly (it's used to fill the differencing buffer).
//
//   For epoch n >= k, transmit the k-th forward difference:
//       d_n = (1-B)^k Y_n   where B is the backward shift operator,
//       i.e. d_n = sum_{i=0..k} (-1)^i C(k,i) Y_{n-i}.
//
// The decoder maintains acc[0..k] where acc[i] is the i-th order
// partial sum. On receiving a delta d, it pushes through:
//
//   acc[k] += d
//   for j from k-1 down to 0:
//     acc[j] += acc[j+1]
//
// acc[0] is the reconstructed value. This is the same algorithm the
// reference `crx2rnx` uses.

#include "crinex_diff.hpp"

namespace rinexpy_native {

void crinex_channel_reset(CrinexChannelState* state, int order) noexcept {
    if (order < 0) order = 0;
    if (order > CrinexChannelState::MAX_ORDER) order = CrinexChannelState::MAX_ORDER;
    state->order = order;
    state->filled = 0;
    for (int i = 0; i < CrinexChannelState::MAX_ORDER; ++i) {
        state->prev[i] = 0;
    }
}

namespace {

// Pascal triangle coefficients C(m, i) for i = 0..m, m in 0..7.
// We need them up to m = MAX_ORDER = 7.
constexpr int PASCAL[8][8] = {
    { 1, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 1, 0, 0, 0, 0, 0, 0 },
    { 1, 2, 1, 0, 0, 0, 0, 0 },
    { 1, 3, 3, 1, 0, 0, 0, 0 },
    { 1, 4, 6, 4, 1, 0, 0, 0 },
    { 1, 5, 10, 10, 5, 1, 0, 0 },
    { 1, 6, 15, 20, 15, 6, 1, 0 },
    { 1, 7, 21, 35, 35, 21, 7, 1 },
};

}  // namespace

std::int64_t crinex_channel_step(CrinexChannelState* state,
                                 std::int64_t delta) noexcept {
    // Pascal-coefficient inverse: the encoder sends Δ^m y_n at
    // epoch n where m = min(n - 1, k). The decoder reconstructs
    //
    //   y_n = delta + sum_{i=1..m} (-1)^(i+1) * C(m, i) * y_{n-i}
    //
    // y_{n-i} is held in state->prev[i-1]. After computing y_n we
    // shift the buffer up (drop the oldest), insert y_n at front.
    const int k = state->order;
    const int m = (state->filled < k) ? state->filled : k;

    std::int64_t y = delta;
    for (int i = 1; i <= m; ++i) {
        const std::int64_t c = PASCAL[m][i];
        if (i & 1) {
            // (-1)^(i+1) = +1 for odd i
            y += c * state->prev[i - 1];
        } else {
            y -= c * state->prev[i - 1];
        }
    }

    // Shift prev down (oldest drops) and insert y at front.
    const int span = (k < CrinexChannelState::MAX_ORDER)
                   ? k : CrinexChannelState::MAX_ORDER;
    for (int i = span - 1; i > 0; --i) {
        state->prev[i] = state->prev[i - 1];
    }
    if (span > 0) state->prev[0] = y;

    if (state->filled < k + 1) state->filled += 1;
    return y;
}

}  // namespace rinexpy_native
