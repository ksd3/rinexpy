// k-th order integer differencing kernel for the CRINEX (Hatanaka)
// compression format.
//
// Reference: Hatanaka, Y. (2008), "A Compression Format and Tools for
// GNSS Observation Data", Bulletin of the Geographical Survey
// Institute, 55, 21-30. Section 3.
//
// For each numeric channel (per-SV, per-observation-type), the CRINEX
// encoder transmits the k-th order forward differences of the integer
// representation of the value. The decoder maintains k accumulators
// and reconstructs the value at each epoch.
//
// This kernel is the per-channel reconstruction primitive. Building
// a full CRINEX -> RINEX decoder on top of it requires the format-
// parsing glue (per-epoch SV lists, init markers, LLI/SSI RLE) that
// the reference `crx2rnx` tool handles; that glue is intentionally
// kept in Python where iterating on edge cases is cheaper.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// State for one numeric channel.
//
// CRINEX uses progressively-higher-order Pascal forward differences:
// epoch 1 transmits the absolute value; epoch n in [2, k+1] transmits
// the (n-1)-th forward difference; epoch n > k+1 transmits the k-th
// forward difference. The decoder mirrors this by tracking the last
// k reconstructed values and applying the matching Pascal-coefficient
// inverse formula at each step.
struct CrinexChannelState {
    static constexpr int MAX_ORDER = 7;
    // prev[0] = v_{n-1}, prev[1] = v_{n-2}, ..., prev[k-1] = v_{n-k}.
    std::int64_t prev[MAX_ORDER] = { 0, 0, 0, 0, 0, 0, 0 };
    int order = 0;          // active differencing order (k)
    int filled = 0;         // how many epochs have been seen (0..k+1)
};

// Reset the channel state. Called when the encoder sends an
// initialization marker (e.g. for a fresh SV or after a reset `&`).
void crinex_channel_reset(CrinexChannelState* state, int order) noexcept;

// Push one transmitted (signed) integer delta through the state and
// return the reconstructed absolute integer value at the current epoch.
// For the first `order` epochs after reset, the delta IS the absolute
// value; thereafter it's a k-th order forward difference.
std::int64_t crinex_channel_step(CrinexChannelState* state,
                                 std::int64_t delta) noexcept;

}  // namespace rinexpy_native
