// RTCM3 CRC-24Q implementation. Table-driven, constexpr-initialised.

#include "crc24q.hpp"

#include <array>

namespace rinexpy_native {

namespace {

constexpr std::uint32_t POLY = 0x1864CFB;

// Build the 256-entry table at compile time so the symbol lives in the
// rodata section and no first-call init overhead pays the cost.
constexpr std::array<std::uint32_t, 256> make_table() noexcept {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t b = 0; b < 256; ++b) {
        std::uint32_t crc = b << 16;
        for (int i = 0; i < 8; ++i) {
            crc <<= 1;
            if (crc & 0x1000000U) {
                crc ^= POLY;
            }
        }
        t[b] = crc & 0xFFFFFFU;
    }
    return t;
}

constexpr auto TABLE = make_table();

}  // namespace

std::uint32_t crc24q(const std::uint8_t* data, std::size_t n) noexcept {
    std::uint32_t crc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t idx = static_cast<std::uint8_t>(
            (crc >> 16) ^ data[i]);
        crc = ((crc << 8) ^ TABLE[idx]) & 0xFFFFFFU;
    }
    return crc;
}

}  // namespace rinexpy_native
