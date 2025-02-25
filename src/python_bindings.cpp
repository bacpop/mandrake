// 2021 John Lees, Gerry Tonkin-Hill, Zhirong Yang
// See LICENSE files

#include <cwchar>
#include <wchar.h>
#include "pairsnp.hpp"
#include "sound.hpp"
#include "wtsne.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
      
PYBIND11_MODULE(SCE, m) {
  m.doc() = "Stochastic cluster embedding";
  m.attr("version") = VERSION_INFO;

  // Results class (need to define here to be able to return this type)
  py::class_<sce_results, std::shared_ptr<sce_results>>(m, "sce_result")
      .def(py::init<const bool, const size_t, const uint64_t>())
      .def("animated", &sce_results::is_animated)
      .def("n_frames", &sce_results::n_frames)
      .def("get_eq", &sce_results::get_eq)
      .def("get_embedding", &sce_results::get_embedding)
      .def("get_embedding_frame", &sce_results::get_embedding_frame,
           py::arg("frame"));

  // Exported functions
  m.def("wtsne", &wtsne, py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding", py::arg("I_vec"), py::arg("J_vec"),
        py::arg("dist_vec"), py::arg("weights"), py::arg("perplexity"),
        py::arg("maxIter"), py::arg("nRepuSamp") = 5, py::arg("eta0") = 1,
        py::arg("bInit") = 0, py::arg("animated") = false,
        py::arg("n_workers") = 128, py::arg("n_threads") = 1,
        py::arg("seed") = 1);

  m.def("pairsnp", &pairsnp, py::return_value_policy::take_ownership,
        "Run pairsnp", py::arg("fasta"), py::arg("n_threads"), py::arg("dist"),
        py::arg("knn"));

  m.def("gen_audio", &sample_wave, py::return_value_policy::take_ownership,
        "Generate audio for animation", py::arg("frequencies"),
        py::arg("duration"), py::arg("sample_rate"), py::arg("n_threads"));

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
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("animated") = false,
        py::arg("cpu_threads") = 1, py::arg("device_id") = 0,
        py::arg("seed") = 1);
  // Use fp32 for single precision (faster, less accurate)
  m.def("wtsne_gpu_fp32", &wtsne_gpu<float>,
        py::return_value_policy::take_ownership,
        "Run stochastic cluster embedding with CUDA", py::arg("I_vec"),
        py::arg("J_vec"), py::arg("dist_vec"), py::arg("weights"),
        py::arg("perplexity"), py::arg("maxIter"), py::arg("blockSize") = 128,
        py::arg("n_workers") = 128, py::arg("nRepuSamp") = 5,
        py::arg("eta0") = 1, py::arg("bInit") = 0, py::arg("animated") = false,
        py::arg("cpu_threads") = 1, py::arg("device_id") = 0,
        py::arg("seed") = 1);
#endif
}
