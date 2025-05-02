// nanobind glue: expose the C++ OBS3 decoder to Python.
//
// The single function exposed is `decode_obs_batch(flat, n_lines, n_obs)`
// returning a numpy float64 array of shape (n_lines, n_obs * 3).
// Signature matches rinexpy._jit.decode_obs_batch so the dispatch code
// in rinexpy.obs3 can swap implementations transparently.

#include "bit_cursor.hpp"
#include "crc24q.hpp"
#include "decode_obs_batch.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace {

// `flat` buffer parameter type: any 1-D contiguous uint8 numpy array.
using BytesArray = nb::ndarray<const std::uint8_t,
                               nb::ndim<1>,
                               nb::c_contig,
                               nb::device::cpu>;

nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::device::cpu>
decode_obs_batch_py(BytesArray flat, std::size_t n_lines, std::size_t n_obs) {
    const std::size_t expected = n_lines * n_obs * rinexpy_native::CELL_WIDTH;
    if (flat.size() < expected) {
        throw std::invalid_argument(
            "flat buffer too small for n_lines * n_obs * 16 cells");
    }

    // Allocate the (n_lines, n_obs * 3) result. nanobind's
    // capsule-managed array transfers ownership to numpy.
    const std::size_t cols = n_obs * 3;
    std::size_t shape[2] = {n_lines, cols};
    double* data = new double[n_lines * cols];

    rinexpy_native::decode_obs_batch(flat.data(), n_lines, n_obs, data);

    // Build the ndarray with a deleter capsule so numpy frees it later.
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });
    return nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::device::cpu>(
        data, 2, shape, owner);
}

}  // namespace

namespace {

// CRC-24Q wrapper: accepts a Python `bytes` and returns the 24-bit
// checksum as a Python int. Matches rinexpy.rtcm3.crc24q.
std::uint32_t crc24q_py(nb::bytes data) {
    const auto* ptr = reinterpret_cast<const std::uint8_t*>(data.c_str());
    return rinexpy_native::crc24q(ptr, data.size());
}

// MSB-first bit extraction. Numerical contract identical to
// rinexpy.rtcm3._bits: unsigned by default, sign-extend when
// is_signed=True. n_bits must be in [0, 64]. Returns a Python int so
// the unsigned-64-bit value range is representable losslessly.
nb::object read_bits_py(nb::bytes data, std::size_t start_bit,
                        unsigned n_bits, bool is_signed) {
    if (n_bits > 64) {
        throw std::invalid_argument("n_bits must be <= 64");
    }
    const auto* ptr = reinterpret_cast<const std::uint8_t*>(data.c_str());
    const std::size_t n = data.size();
    if (is_signed) {
        const std::int64_t s = rinexpy_native::read_bits_signed(
            ptr, n, start_bit, n_bits);
        return nb::int_(s);
    }
    const std::uint64_t u = rinexpy_native::read_bits(
        ptr, n, start_bit, n_bits);
    return nb::int_(u);
}

}  // namespace

NB_MODULE(_ext, m) {
    m.doc() = "Internal C++ acceleration for rinexpy.obs3 and rinexpy.rtcm3.";

    m.def(
        "decode_obs_batch",
        &decode_obs_batch_py,
        nb::arg("flat"),
        nb::arg("n_lines"),
        nb::arg("n_obs"),
        "Decode N concatenated SV observation lines into a (N, n_obs*3) "
        "float64 array.\n\n"
        "Drop-in replacement for rinexpy._jit.decode_obs_batch with the "
        "same numerical contract: each cell is (value, LLI, SSI); empty "
        "cells become NaN.");

    m.def(
        "crc24q",
        &crc24q_py,
        nb::arg("data"),
        "RTCM3 CRC-24Q over `data`. Polynomial 0x1864CFB, init 0, no\n"
        "reflection, no final XOR. Returns the 24-bit checksum.");

    m.def(
        "read_bits",
        &read_bits_py,
        nb::arg("data"),
        nb::arg("start_bit"),
        nb::arg("n_bits"),
        nb::arg("is_signed") = false,
        "MSB-first bit extraction. Reads `n_bits` (<=64) from `data` at\n"
        "bit offset `start_bit`. When `is_signed=True`, sign-extends.\n"
        "Bit-identical to rinexpy.rtcm3._bits.");
}
