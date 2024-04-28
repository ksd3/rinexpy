// Header for the OBS3 fixed-width decoder kernel.
//
// The kernel takes a flat ASCII byte buffer holding `n_lines` SV
// observation lines, each padded to `n_obs * 16` bytes (14-byte value
// + 1-byte LLI + 1-byte SSI per measurement), and decodes it into a
// (n_lines, n_obs * 3) float64 array laid out [val0, lli0, ssi0,
// val1, lli1, ssi1, ...] per row.
//
// The numerical contract matches rinexpy.obs3._decode_sv_line and the
// numba kernel rinexpy._jit.decode_obs_batch bit-for-bit so the
// caller can A/B compare.

#pragma once

#include <cstddef>
#include <cstdint>

namespace rinexpy_native {

// Width constants per RINEX 3 OBS spec (DF398 etc.).
inline constexpr std::size_t FIELD_WIDTH = 14;
inline constexpr std::size_t CELL_WIDTH = 16;

// Decode one cell at byte offset `start` in `buf` and write the three
// (value, LLI, SSI) outputs into `out_row` starting at `col * 3`.
// Empty value cells produce NaN; non-digit LLI/SSI bytes also become
// NaN.
void decode_one_cell(const std::uint8_t* buf,
                     std::size_t start,
                     double* out_row,
                     std::size_t col) noexcept;

// Decode `n_lines * n_obs * 3` doubles into `out`. Caller must size
// `out` appropriately. `flat` must be at least `n_lines * n_obs *
// CELL_WIDTH` bytes long (no bounds check).
void decode_obs_batch(const std::uint8_t* flat,
                      std::size_t n_lines,
                      std::size_t n_obs,
                      double* out) noexcept;

}  // namespace rinexpy_native
