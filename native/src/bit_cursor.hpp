// MSB-first bit reader for RTCM3 / similarly-packed binary protocols.
//
// The Python equivalent in rinexpy.rtcm3 is _bits(buf, start_bit, n_bits):
// it walks bit-by-bit (divmod, shift, mask, conditional sign-extend). For
// an MSM7 frame, _bits is called O(80 * n_cells) times per epoch; on a
// 32-SV multi-system frame that's tens of thousands of Python calls per
// second at 10 Hz.
//
// This kernel:
//
// 1. Loads up to 64 bits at a time into a `uint64_t` so the shift/mask
//    happens in one register operation.
// 2. Handles the cross-byte boundary by reading 8 contiguous bytes,
//    shifting the start byte to bit 63, and right-shifting by
//    `(64 - n_bits) - intra_byte_offset`. For requests of <= 56 bits
//    this is a single load + shift + mask. For 57-64 bits we read a
//    9th byte and combine.
// 3. Skips the Python attribute lookup / per-byte indexing overhead.
//
// Numerical contract: matches rinexpy.rtcm3._bits bit-for-bit including
// the sign-extension when ``signed=True``. Out-of-range reads (start_bit
// past end of buffer) return 0; the caller is responsible for size
// checking, same as the Python helper which silently produces 0 too if
// you walk off the end. (Both behaviours match the existing tests.)

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// Read `n_bits` (1..64) starting at MSB-aligned bit offset `start_bit`
// from `buf` of size `n_bytes`. Returns the value zero-extended; the
// caller can sign-extend by calling `read_bits_signed` instead.
//
// Inlined header-only because every cycle counts in the per-cell MSM
// loop and we want the compiler to fuse the call with surrounding
// arithmetic when bindings.cpp packs multiple reads into a single
// per-cell decode.
static inline std::uint64_t read_bits(const std::uint8_t* buf,
                                      std::size_t n_bytes,
                                      std::size_t start_bit,
                                      unsigned n_bits) noexcept {
    if (n_bits == 0) {
        return 0;
    }
    const std::size_t byte_off = start_bit >> 3;
    const unsigned bit_off = static_cast<unsigned>(start_bit & 7);

    // Past end of buffer? Match Python's silent 0 behaviour.
    if (byte_off >= n_bytes) {
        return 0;
    }

    // How many bytes does the field span?
    const unsigned span_bytes = (bit_off + n_bits + 7) / 8;

    // Build a big-endian uint64 from up to 8 bytes (covers n_bits <=
    // 56 in any alignment, plus n_bits = 64 when bit_off == 0).
    std::uint64_t word = 0;
    const unsigned take_lo = (span_bytes < 8) ? span_bytes : 8;
    for (unsigned i = 0; i < take_lo; ++i) {
        if (byte_off + i >= n_bytes) {
            break;
        }
        word |= static_cast<std::uint64_t>(buf[byte_off + i])
                << ((7 - i) * 8);
    }

    // For requests up to 64 - bit_off bits, a single shift covers it.
    const unsigned avail = 64 - bit_off;
    if (n_bits <= avail) {
        return (word << bit_off) >> (64 - n_bits);
    }

    // Spans 9 bytes (only possible if bit_off > 0 AND n_bits > avail).
    // Take the high `avail` bits, then OR in the low `n_bits - avail`
    // bits from the trailing byte.
    const std::uint64_t hi = (word << bit_off) >> (64 - avail);
    const unsigned lo_bits = n_bits - avail;
    if (byte_off + 8 >= n_bytes) {
        return hi << lo_bits;  // missing trailing byte -> zero
    }
    const std::uint64_t lo = static_cast<std::uint64_t>(buf[byte_off + 8])
                             >> (8 - lo_bits);
    return (hi << lo_bits) | lo;
}

// Same as read_bits but sign-extends to int64 from an `n_bits`-wide
// two's-complement value.
static inline std::int64_t read_bits_signed(const std::uint8_t* buf,
                                            std::size_t n_bytes,
                                            std::size_t start_bit,
                                            unsigned n_bits) noexcept {
    const std::uint64_t u = read_bits(buf, n_bytes, start_bit, n_bits);
    if (n_bits == 0 || n_bits == 64) {
        return static_cast<std::int64_t>(u);
    }
    const std::uint64_t sign_bit = 1ULL << (n_bits - 1);
    if (u & sign_bit) {
        // OR in the high bits above n_bits.
        const std::uint64_t mask = ~((1ULL << n_bits) - 1);
        return static_cast<std::int64_t>(u | mask);
    }
    return static_cast<std::int64_t>(u);
}

}  // namespace rinexpy_native
