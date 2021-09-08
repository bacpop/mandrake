// 2020 John Lees and Gerry Tonkin-Hill

#include "wtsne.hpp"

PYBIND11_MODULE(SCE, m) {
  m.doc() = "Stochastic cluster embedding";
  m.attr("version") = VERSION_INFO;

  // Exported functions
  m.def("wtsne", &wtsne, py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding", py::arg("I_vec"), py::arg("J_vec"),
        py::arg("dist_vec"), py::arg("weights"), py::arg("perplexity"),
        py::arg("maxIter"), py::arg("nRepuSamp") = 5, py::arg("eta0") = 1,
        py::arg("bInit") = 0, py::arg("n_workers") = 128, py::arg("n_threads") = 1, py::arg("seed") = 1);

#ifdef GPU_AVAILABLE
  // NOTE: python always uses fp64 so cannot easily template these (which
  // would just give one function name exported but with different type
  // prototypes). To do this would require numpy (python)/Eigen (C++) which
  // support both precisions. But easier just to stick with List (python)/
  // std::vector (C++) and allow fp64->fp32 conversion when called

  // Use fp64 for double precision (slower, more accurate)
  m.def("wtsne_gpu_fp64", &wtsne_gpu<double>,
        py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding with CUDA", py::arg("I_vec"),
        py::arg("J_vec"), py::arg("dist_vec"), py::arg("weights"),
        py::arg("perplexity"), py::arg("maxIter"), py::arg("blockSize") = 128,
        py::arg("n_workers") = 128, py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("n_threads") = 1,
        py::arg("deviceId") = 0, py::arg("seed") = 1);
  // Use fp32 for single precision (faster, less accurate)
  m.def("wtsne_gpu_fp32", &wtsne_gpu<float>, py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding with CUDA", py::arg("I_vec"),
        py::arg("J_vec"), py::arg("dist_vec"), py::arg("weights"),
        py::arg("perplexity"), py::arg("maxIter"), py::arg("blockSize") = 128,
        py::arg("n_workers") = 128, py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("n_threads") = 1,
        py::arg("deviceId") = 0, py::arg("seed") = 1);
#endif
}
