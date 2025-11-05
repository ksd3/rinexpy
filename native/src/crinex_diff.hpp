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

// State for one numeric channel. Holds the running k accumulators.
struct CrinexChannelState {
    // accumulator[i] is the i-th order partial sum. Order k is set by
    // the caller and bounded to 7 (CRINEX never exceeds order 5 in
    // practice; we keep some headroom).
    static constexpr int MAX_ORDER = 7;
    std::int64_t acc[MAX_ORDER + 1] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int order = 0;          // active differencing order
    int filled = 0;         // how many epochs have been seen
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
