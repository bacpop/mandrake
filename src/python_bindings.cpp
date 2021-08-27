// 2020 John Lees and Gerry Tonkin-Hill

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "wtsne.hpp"

PYBIND11_MODULE(SCE, m) {
  m.doc() = "Stochastic cluster embedding";
  m.attr("version") = VERSION_INFO;

  // Exported functions
  m.def("wtsne", &wtsne, py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding", py::arg("I_vec"), py::arg("J_vec"),
        py::arg("dist_vec"), py::arg("weights"), py::arg("perplexity"),
        py::arg("maxIter"), py::arg("nRepuSamp") = 5, py::arg("eta0") = 1,
        py::arg("bInit") = 0, py::arg("n_threads") = 1, py::arg("seed") = 1);

#ifdef GPU_AVAILABLE
  m.def("wtsne_gpu", &wtsne_gpu<double>,
        py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding with CUDA", py::arg("I_vec"),
        py::arg("J_vec"), py::arg("dist_vec"), py::arg("weights"),
        py::arg("perplexity"), py::arg("maxIter"), py::arg("blockSize") = 128,
        py::arg("blockCount") = 128, py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("n_threads") = 1,
        py::arg("deviceId") = 0,, py::arg("seed") = 1);
  m.def("wtsne_gpu", &wtsne_gpu<float>, py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding with CUDA", py::arg("I_vec"),
        py::arg("J_vec"), py::arg("dist_vec"), py::arg("weights"),
        py::arg("perplexity"), py::arg("maxIter"), py::arg("blockSize") = 128,
        py::arg("blockCount") = 128, py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("n_threads") = 1,
        py::arg("deviceId") = 0, py::arg("seed") = 1);
#endif
}
