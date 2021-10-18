// 2021 John Lees, Gerry Tonkin-Hill, Zhirong Yang
// See LICENSE files

#include "pairsnp.hpp"
#include "wtsne.hpp"

#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(SCE, m) {
  m.doc() = "Stochastic cluster embedding";
  m.attr("version") = VERSION_INFO;

  // Results class (need to define here to be able to return this type)
  py::class_<sce_results<double>, std::shared_ptr<sce_results<double>>>(
      m, "sce_result")
      .def(py::init<const bool, const size_t, const uint64_t>())
      .def("animated", &sce_results<double>::is_animated)
      .def("n_frames", &sce_results<double>::n_frames)
      .def("get_eq", &sce_results<double>::get_eq)
      .def("get_embedding", &sce_results<double>::get_embedding)
      .def("get_embedding_frame", &sce_results<double>::get_embedding_frame,
           py::arg("frame"))
      // TODO - do this with a smart pointer instead
      .def(py::pickle(
        [](const sce_results<double> &results) {
          return py::make_tuple(p.is_animated(), p.get_eq(), p.get_all_embeddings());
        },
        [](py::tuple t) {
          if (t.size() != 3) {
            throw std::runtime_error("Invalid state during pickle of SCE results")
          }
          sce_results results(t[0], t[1], t[2]);
          return results;
        }
      ));

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
