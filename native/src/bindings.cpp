// nanobind glue: expose the C++ OBS3 decoder to Python.
//
// The single function exposed is `decode_obs_batch(flat, n_lines, n_obs)`
// returning a numpy float64 array of shape (n_lines, n_obs * 3).
// Signature matches rinexpy._jit.decode_obs_batch so the dispatch code
// in rinexpy.obs3 can swap implementations transparently.

#include "bit_cursor.hpp"
#include "crc24q.hpp"
#include "decode_obs_batch.hpp"
#include "lambda_ils.hpp"
#include "msm_decode.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

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

// Helper: build a nanobind-owned 1-D / 2-D float64 / int64 ndarray
// over a fresh heap allocation, with a deleter capsule so Python owns it.
template <typename T>
nb::ndarray<nb::numpy, T, nb::device::cpu>
make_owned_1d(const std::vector<T>& v) {
    T* data = new T[v.size()];
    std::memcpy(data, v.data(), v.size() * sizeof(T));
    std::size_t shape[1] = { v.size() };
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<T*>(p);
    });
    return nb::ndarray<nb::numpy, T, nb::device::cpu>(data, 1, shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::device::cpu>
make_owned_2d(const std::vector<T>& v, std::size_t rows, std::size_t cols) {
    T* data = new T[v.size()];
    std::memcpy(data, v.data(), v.size() * sizeof(T));
    std::size_t shape[2] = { rows, cols };
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<T*>(p);
    });
    return nb::ndarray<nb::numpy, T, nb::device::cpu>(data, 2, shape, owner);
}

// LAMBDA ILS wrapper. Returns (candidates int64 (k, n), sq_errors
// float64 (k,), nodes_visited, aborted_reason). aborted_reason: 0 OK,
// 1 max_nodes, 2 max_seconds. The Python lambda_ar layer translates
// that into ILSAborted.
nb::tuple lambda_ils_py(
        nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a_float,
        nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu> Q,
        std::size_t n_cands,
        std::uint64_t max_nodes,
        double max_seconds) {
    const std::size_t n = a_float.size();
    if (Q.shape(0) != n || Q.shape(1) != n) {
        throw std::invalid_argument(
            "lambda_ils: Q must be square (n, n) matching a_float length");
    }
    rinexpy_native::IlsResult res = rinexpy_native::integer_least_squares(
        a_float.data(), Q.data(), n, n_cands, max_nodes, max_seconds);

    auto cands_arr = make_owned_2d<std::int64_t>(
        res.candidates, res.n_returned, n);
    auto sq_arr = make_owned_1d<double>(res.sq_errors);
    auto L_arr = make_owned_2d<double>(res.L_factor, n, n);

    return nb::make_tuple(cands_arr, sq_arr, L_arr,
                          static_cast<std::uint64_t>(res.nodes_visited),
                          res.aborted_reason);
}

// Full MSM4 / MSM7 frame decoder. Returns a Python dict with the
// header scalars plus several typed ndarrays:
//
//   - sv_indices       int32 (n_sv,)
//   - signal_indices   int32 (n_sig,)
//   - cell_mask        uint8 (n_sv * n_sig,)
//   - rough_range_ms   float64 (n_sv,)
//   - extended_info    int32 (n_sv,)
//   - rough_doppler    int32 (n_sv,)        raw signed-14-bit value
//   - obs_sv_k         int32 (n_present,)   index into sv_indices
//   - obs_sig_k        int32 (n_present,)   index into signal_indices
//   - pseudorange_m    float64 (n_present,)
//   - phase_m          float64 (n_present,)
//   - lock_time        int32 (n_present,)
//   - half_cycle_ambiguity int32 (n_present,)
//   - cnr_dbhz         float64 (n_present,)
//   - doppler_mps      float64 (n_present,) NaN for MSM4 cells
//   - payload_truncated bool
//
// The Python wrapper assembles the public dict-of-list-of-dicts shape
// from these arrays so existing callers (test_rtcm3_real, NTRIP
// streamers, real-time PPP) don't have to change.
nb::dict decode_msm_py(nb::bytes body, int msm_kind) {
    if (msm_kind != 4 && msm_kind != 7) {
        throw std::invalid_argument("msm_kind must be 4 or 7");
    }
    const auto* ptr = reinterpret_cast<const std::uint8_t*>(body.c_str());
    const std::size_t n = body.size();
    rinexpy_native::MsmResult r = rinexpy_native::decode_msm(
        ptr, n, msm_kind);

    nb::dict d;
    d["station_id"] = r.station_id;
    d["tow_ms"] = r.tow_ms;
    d["sync"] = r.sync;
    d["iod"] = r.iod;
    d["smoothing_indicator"] = r.smoothing_indicator;
    d["smoothing_interval"] = r.smoothing_interval;
    d["sv_mask"] = r.sv_mask;
    d["signal_mask"] = r.signal_mask;
    d["n_sv"] = r.n_sv;
    d["n_sig"] = r.n_sig;
    d["payload_truncated"] = r.payload_truncated;

    // Convert vector<int> -> int32 ndarray.
    auto i32 = [](const std::vector<int>& v) {
        std::int32_t* data = new std::int32_t[v.size() ? v.size() : 1];
        for (std::size_t i = 0; i < v.size(); ++i) {
            data[i] = static_cast<std::int32_t>(v[i]);
        }
        std::size_t shape[1] = { v.size() };
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<std::int32_t*>(p);
        });
        return nb::ndarray<nb::numpy, std::int32_t, nb::device::cpu>(
            data, 1, shape, owner);
    };
    auto u8 = [](const std::vector<std::uint8_t>& v) {
        std::uint8_t* data = new std::uint8_t[v.size() ? v.size() : 1];
        std::memcpy(data, v.data(), v.size());
        std::size_t shape[1] = { v.size() };
        nb::capsule owner(data, [](void* p) noexcept {
            delete[] static_cast<std::uint8_t*>(p);
        });
        return nb::ndarray<nb::numpy, std::uint8_t, nb::device::cpu>(
            data, 1, shape, owner);
    };
    auto f64 = [](const std::vector<double>& v) {
        return make_owned_1d<double>(v);
    };

    d["sv_indices"] = i32(r.sv_indices);
    d["signal_indices"] = i32(r.signal_indices);
    d["cell_mask"] = u8(r.cell_mask);
    d["rough_range_ms"] = f64(r.rough_range_ms);
    d["extended_info"] = i32(r.extended_info);
    d["rough_doppler"] = i32(r.rough_doppler);
    d["obs_sv_k"] = i32(r.obs_sv_k);
    d["obs_sig_k"] = i32(r.obs_sig_k);
    d["pseudorange_m"] = f64(r.pseudorange_m);
    d["phase_m"] = f64(r.phase_m);
    d["lock_time"] = i32(r.lock_time);
    d["half_cycle_ambiguity"] = i32(r.half_cycle_ambiguity);
    d["cnr_dbhz"] = f64(r.cnr_dbhz);
    d["doppler_mps"] = f64(r.doppler_mps);

    return d;
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

    m.def(
        "lambda_ils",
        &lambda_ils_py,
        nb::arg("a_float"),
        nb::arg("Q"),
        nb::arg("n_cands"),
        nb::arg("max_nodes"),
        nb::arg("max_seconds"),
        "LAMBDA branch-and-bound integer least squares. Returns\n"
        "(candidates int64 (k,n), sq_errors float64 (k,), nodes,\n"
        "aborted_reason in {0,1,2}).");

    m.def(
        "decode_msm",
        &decode_msm_py,
        nb::arg("body"),
        nb::arg("msm_kind"),
        "Decode an MSM4 (msm_kind=4) or MSM7 (msm_kind=7) RTCM3 frame\n"
        "body. Returns a dict of header scalars plus several typed\n"
        "ndarrays (sv_indices, signal_indices, cell_mask, the per-SV\n"
        "block, and parallel arrays of per-cell observations).");
}
