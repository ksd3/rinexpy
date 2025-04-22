// RTCM3 CRC-24Q kernel.
//
// Polynomial 0x1864CFB, init 0, no reflection, no final XOR. Used to
// trail every RTCM3 frame (preamble + 2-byte length + body + CRC).
//
// We precompute a 256-entry lookup table at static-init time so the
// per-byte cost collapses to a single XOR + shift + table read; the
// pure-Python equivalent in rinexpy.rtcm3 does eight conditional XORs
// per byte plus a per-byte Python attribute fetch.
//
// Reentrant. No global mutable state except the static const table.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// Compute the CRC-24Q of `data[0 .. n)` starting from initial CRC 0.
// Returns a 24-bit value in the low bits of the uint32 (high byte 0).
std::uint32_t crc24q(const std::uint8_t* data, std::size_t n) noexcept;

}  // namespace rinexpy_native
