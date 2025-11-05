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
    for (int i = 0; i <= CrinexChannelState::MAX_ORDER; ++i) {
        state->acc[i] = 0;
    }
}

std::int64_t crinex_channel_step(CrinexChannelState* state,
                                 std::int64_t delta) noexcept {
    // The RTKLIB / reference algorithm: shift the buffer down, write
    // the new delta into slot 0, then cumulative-sum across the
    // buffer to produce the reconstructed value (which lands in
    // slot 0). On the next iteration that slot shifts to slot 1, etc.
    const int k = state->order;
    for (int i = k; i > 0; --i) {
        state->acc[i] = state->acc[i - 1];
    }
    state->acc[0] = delta;
    for (int i = 1; i <= k; ++i) {
        state->acc[0] += state->acc[i];
    }
    state->filled += 1;
    return state->acc[0];
}

}  // namespace rinexpy_native
