// GPT2w cell evaluator.
//
// rinexpy.gpt2w.gpt2w evaluates 8 quantities (pressure, temperature,
// humidity, lapse rate, mean temperature, water-vapour decrease,
// VMF1 a_h and a_w) at a query (lat, lon, doy) by bilinearly
// interpolating across 4 surrounding grid cells. Each cell stores
// a mean + 4 seasonal coefficients per quantity, so each per-cell
// evaluation does 4 trig calls + 5 multiplies for each of 8
// quantities = ~32 trig calls per cell, ~128 per gpt2w call.
//
// The seasonal-trig terms (cos/sin of annual + semi-annual phase)
// depend ONLY on doy and are identical across the 4 corners. The
// Python reference re-derives them per corner; this kernel computes
// them once and runs the rest as scalar arithmetic.

#pragma once

namespace rinexpy_native {

// Evaluate GPT2w at one query point.
//
// `cells`      (4 * 44,) flat float64. Cells in row-major order:
//              [c00, c01, c10, c11] of the surrounding grid square,
//              each cell laid out exactly like the Python `data[i,j]`
//              row (undulation, ref_h, then 8 groups of 5 seasonal
//              coefs).
// `w_lat`,
// `w_lon`      bilinear weights in [0, 1].
// `doy`        day of year.
// `altitude_m` receiver altitude (used for the altitude reduction).
// `out`        7-element float64 output:
//              [pressure_hpa, temperature_k, e_hpa, a_h, a_w,
//               T_lapse, undulation_m]
void gpt2w_eval_cell(const double* cells, double w_lat, double w_lon,
                     double doy, double altitude_m,
                     double* out) noexcept;

}  // namespace rinexpy_native
