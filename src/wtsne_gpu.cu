/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Weighted t-SNE with stochastic optimization
 ============================================================================
 */

// Modified by John Lees 2021

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#include "containers.cuh"
#include "cuda_launch.cuh"
#include "uniform_discrete.hpp"
#include "wtsne.hpp"

/****************************
 * Kernels                  *
 ****************************/

template <typename real_t>
KERNEL void destride_embedding(real_t* Y_interleaved, real_t* Y_blocked, size_t size, size_t n_samples) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
    idx += blockDim.x * gridDim.x) {
    int i = idx / DIM;
    int j = idx % DIM;
    Y_blocked[i * DIM + j] = Y_interleaved[i + j * n_samples];
  }
}

// Updates the embedding
// NB: strides of Y are switched in the GPU code compared to the CPU code
template <typename real_t>
KERNEL void wtsneUpdateYKernel(uint32_t *rng_state,
                               const discrete_table_ptrs<real_t> node_table,
                               const discrete_table_ptrs<real_t> edge_table,
                               volatile real_t *Y, uint64_t *I, uint64_t *J,
                               real_t *Eq, real_t *qsum, uint64_t *qcount,
                               uint64_t nn, uint64_t ne, real_t eta0,
                               uint64_t nRepuSamp, real_t nsq, bool bInit,
                               uint64_t *iter, uint64_t maxIter, int n_workers,
                               unsigned long long int *clash_cnt) {
  // Worker index based on CUDA launch parameters
  int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // Iteration parameters
  const real_t one = 1.0; // Used a few times, fp64/fp32
  real_t eta = eta0 * (1 - static_cast<real_t>(*iter) / (maxIter - 1));
  eta = MAX(eta, eta0 * 1e-4);
  real_t attrCoef = (bInit && *iter < maxIter / 10) ? 8 : 2;

  if (workerIdx < n_workers) {
    // Bring RNG state into local registers
    interleaved<uint32_t> p_rng(rng_state, workerIdx, n_workers);
    rng_state_t<real_t> rng_block = get_rng_state<real_t>(p_rng);

    real_t dY[DIM];
    real_t Yk_read[DIM];
    real_t Yl_read[DIM];
    real_t c = one / ((*Eq) * nsq);

    real_t qsum_local = 0.0;
    uint64_t qcount_local = 0;

    real_t repuCoef = 2 * c / nRepuSamp * nsq;
    for (int r = 0; r < nRepuSamp + 1; r++) {
      uint64_t k, l;
      if (r == 0) {
        uint64_t e = discrete_draw(rng_block, edge_table) % ne;
        k = I[e];
        l = J[e];
      } else {
        k = discrete_draw(rng_block, node_table) % nn;
        l = discrete_draw(rng_block, node_table) % nn;
      }

      if (k != l) {
        real_t dist2 = static_cast<real_t>(0.0);
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          // These are read here to avoid multiple workers writing to the same
          // location below
          Yk_read[d] = Y[k + d * nn];
          Yl_read[d] = Y[l + d * nn];
          dY[d] = Yk_read[d] - Yl_read[d];
          dist2 += dY[d] * dY[d];
        }
        __threadfence();

        real_t q = one / (1 + dist2);

        real_t g;
        if (r == 0) {
          g = -attrCoef * q;
        } else {
          g = repuCoef * q * q;
        }

        bool overwrite = false;
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          real_t gain = eta * g * dY[d];
          // The atomics below basically do
          // Y[d + lk] += gain;
          // Y[d + ll] -= gain;
          if (atomicAdd((real_t *)Y + k + d * nn, gain) != Yk_read[d] ||
              atomicAdd((real_t *)Y + l + d * nn, -gain) != Yl_read[d]) {
            overwrite = true;
          }
        }
        qsum_local += q;
        qcount_local++;
        if (overwrite) {
          atomicAdd(clash_cnt, 1ULL);
        }
      }
    }
    __syncwarp();

    // Store local state (RNG & counts) back to global
    put_rng_state(rng_block, p_rng);
    qsum[workerIdx] = qsum_local;
    qcount[workerIdx] = qcount_local;
  }
}

/****************************
 * Classes                  *
 ****************************/
template <typename real_t> struct kernel_ptrs {
  uint32_t *rng;
  real_t *Y;
  uint64_t *I;
  uint64_t *J;
  real_t *Eq;
  real_t *qsum;
  uint64_t *qcount;
  uint64_t nn;
  uint64_t ne;
  real_t nsq;
  int n_workers;
};

template <typename real_t> struct callBackData_t {
  real_t *Eq;
  real_t *nsq;
  real_t *qsum;
  uint64_t *qcount;
  real_t *eta0;
  int *n_write;
  unsigned long long int *n_clashes;
  uint64_t *iter;
  uint64_t *maxIter;
};

// Callback, which is a CUDA host function that updates the progress meter
// and calculates Eq
template <typename real_t> void CUDART_CB Eq_callback(void *data) {
  callBackData_t<real_t> *tmp = (callBackData_t<real_t> *)(data);
  real_t *Eq = tmp->Eq;
  real_t *nsq = tmp->nsq;
  real_t *qsum = tmp->qsum;
  uint64_t *qcount = tmp->qcount;
  *Eq = (*Eq * *nsq + *qsum) / (*nsq + *qcount);

  uint64_t *iter = tmp->iter;
  uint64_t *maxIter = tmp->maxIter;
  if (*iter % MAX(1, *maxIter / 1000) == 0) {
    real_t *eta0 = tmp->eta0;
    real_t eta = *eta0 * (1 - static_cast<real_t>(*iter) / (*maxIter - 1));
    eta = MAX(eta, *eta0 * 1e-4);

    int *n_write = tmp->n_write;
    unsigned long long int *n_clashes = tmp->n_clashes;
    update_progress(*iter, *maxIter, eta, *Eq, *n_write, *n_clashes);
  }
}

// This is the class that does all the work
template <typename real_t> class sce_gpu {
public:
  sce_gpu(const std::vector<real_t> &Y, const std::vector<uint64_t> &I,
          const std::vector<uint64_t> &J, const std::vector<double> &P,
          const std::vector<real_t> &weights, int n_workers,
          const int device_id, const unsigned int seed)
      : n_workers_(n_workers), nn_(weights.size()), ne_(P.size()),
        nsq_(static_cast<real_t>(nn_) * (nn_ - 1)),
        progress_callback_fn_(Eq_callback<real_t>),
        rng_state_(load_rng<real_t>(n_workers, seed)), Y_(Y), Y_destride_(Y.size()),
        Y_host_(Y), I_(I), J_(J),
        Eq_host_(1.0), Eq_device_(1.0), qsum_(n_workers), qsum_total_host_(0.0),
        qsum_total_device_(0.0), qcount_(n_workers), qcount_total_host_(0),
        qcount_total_device_(0) {
    // Initialise CUDA
    CUDA_CALL(cudaSetDevice(device_id));
#ifdef USE_CUDA_PROFILER
    CUDA_CALL(cudaProfilerStart());
#endif

    // Initialise tmp space for reductions on qsum and qcount
    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                           qsum_.data(), qsum_total_device_.data(),
                           qsum_.size());
    qsum_tmp_storage_.set_size(qsum_tmp_storage_bytes_);
    cub::DeviceReduce::Sum(qcount_tmp_storage_.data(),
                           qcount_tmp_storage_bytes_, qcount_.data(),
                           qcount_total_device_.data(), qcount_.size());
    qcount_tmp_storage_.set_size(qcount_tmp_storage_bytes_);

    // Set up discrete RNG tables
    node_table_ = set_device_table(weights);
    edge_table_ = set_device_table(P);
  }

#ifdef USE_CUDA_PROFILER
  ~sce_gpu() { CUDA_CALL_NOTHROW(cudaProfilerStop()); }
#endif

  real_t current_Eq() const { return Eq_host_; }

  // This runs the SCE loop on the device
  void run_SCE(std::shared_ptr<sce_results<real_t>> results,
               uint64_t maxIter, const int block_size, const int n_workers,
               const uint64_t nRepuSamp, real_t eta0, const bool bInit) {
    using namespace std::literals;

    uint64_t iter_h = 0;
    device_value<uint64_t> iter_d(iter_h);
    const int write_per_worker = n_workers * (nRepuSamp + 1);
    unsigned long long int n_clashes_h = 0;
    device_value<unsigned long long int> n_clashes_d(n_clashes_h);
    kernel_ptrs<real_t> device_ptrs = get_device_ptrs();

    // Save the starting positions
    real_t curr_Eq = Eq_host_;
    uint64_t curr_iter = iter_h;
    results->add_frame(curr_iter, curr_Eq, Y_host_);

    // Set up a single iteration on a CUDA graph
    const size_t block_count = (n_workers_ + block_size - 1) / block_size;
    cuda_graph graph;
    cuda_stream capture_stream, copy_stream, graph_stream;

    // Set up pointers used for kernel parameters in graph
    progress_callback_params_.Eq = &Eq_host_;
    progress_callback_params_.nsq = &nsq_;
    progress_callback_params_.qsum = &qsum_total_host_;
    progress_callback_params_.qcount = &qcount_total_host_;
    progress_callback_params_.eta0 = &eta0;
    progress_callback_params_.n_write = &write_per_worker;
    progress_callback_params_.n_clashes = &n_clashes_h;
    progress_callback_params_.iter = &iter_h;
    progress_callback_params_.maxIter = &maxIter;

    // SCE updates kernel with workers, then updates Eq
    // Start capture
    capture_stream.capture_start();
    iter_d.set_value_async(&iter_h, capture_stream.stream());
    wtsneUpdateYKernel<real_t>
        <<<block_count, block_size, 0, capture_stream.stream()>>>(
            device_ptrs.rng, get_node_table(), get_edge_table(), device_ptrs.Y,
            device_ptrs.I, device_ptrs.J, device_ptrs.Eq, device_ptrs.qsum,
            device_ptrs.qcount, device_ptrs.nn, device_ptrs.ne, eta0, nRepuSamp,
            device_ptrs.nsq, bInit, iter_d.data(), maxIter,
            device_ptrs.n_workers, n_clashes_d.data());

    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                           qsum_.data(), qsum_total_device_.data(),
                           qsum_.size(), capture_stream.stream());
    cub::DeviceReduce::Sum(
        qcount_tmp_storage_.data(), qcount_tmp_storage_bytes_, qcount_.data(),
        qcount_total_device_.data(), qcount_.size(), capture_stream.stream());
    qsum_total_device_.get_value_async(&qsum_total_host_,
                                       capture_stream.stream());
    qcount_total_device_.get_value_async(&qcount_total_host_,
                                         capture_stream.stream());
    n_clashes_d.get_value_async(&n_clashes_h, capture_stream.stream());

    capture_stream.add_host_fn(progress_callback_fn_,
                               (void *)&progress_callback_params_);
    Eq_device_.set_value_async(&Eq_host_, capture_stream.stream());

    capture_stream.capture_end(graph.graph());
    // End capture

    // Main SCE loop - run captured graph maxIter times
    // NB: Here I have written the code so the kernel launch parameters (and all
    // CUDA API calls) are able to use the same parameters each loop, mainly by
    // using pointers to device memory, and two iter counters.
    // The alternative would be to use cudaGraphExecKernelNodeSetParams to
    // change the kernel launch parameters. See
    // 0c369b209ef69d91016bedd41ea8d0775879f153
    const auto start = std::chrono::steady_clock::now();
    for (iter_h = 0; iter_h < maxIter; ++iter_h) {
      graph.launch(graph_stream.stream());
      if (iter_h % MAX(1, maxIter / 1000) == 0) {
        check_interrupts();
      }
      if (results->is_sample_frame(iter_h)) {
        update_frames(results, graph_stream, copy_stream, curr_iter, curr_Eq, iter_h, Eq_host_);
      }
    }
    graph_stream.sync();
    const auto end = std::chrono::steady_clock::now();

    if (results->n_frames() > 0) {
      // Save penultimate frame
      update_frames(results, graph_stream, copy_stream, curr_iter, curr_Eq, maxIter, Eq_host_);
      // Save final frame
      copy_stream.sync();
      results->add_frame(maxIter, Eq_host_, Y_host_);
    }
    std::cerr << std::endl << "Optimizing done in " << (end - start) / 1s << "s" << std::endl;
  }

  // Copy result back to host
  std::vector<real_t>& get_embedding_result() {
    cuda_stream stream;
    save_embedding_result(stream, stream);
    stream.sync();
    return Y_host_;
  }

private:
  template <typename T>
  discrete_table_device<real_t> set_device_table(const std::vector<T> &probs) {
    discrete_table<real_t, T> table(probs);
    discrete_table_device<real_t> dev_table = {.F = table.F_table(),
                                               .A = table.A_table()};
    return dev_table;
  }

  discrete_table_ptrs<real_t> get_node_table() {
    discrete_table_ptrs<real_t> device_node_table = {.K = node_table_.F.size(),
                                                     .F = node_table_.F.data(),
                                                     .A = node_table_.A.data()};
    return device_node_table;
  }

  discrete_table_ptrs<real_t> get_edge_table() {
    discrete_table_ptrs<real_t> device_edge_table = {.K = edge_table_.F.size(),
                                                     .F = edge_table_.F.data(),
                                                     .A = edge_table_.A.data()};
    return device_edge_table;
  }

  kernel_ptrs<real_t> get_device_ptrs() {
    kernel_ptrs<real_t> device_ptrs = {.rng = rng_state_.data(),
                                       .Y = Y_.data(),
                                       .I = I_.data(),
                                       .J = J_.data(),
                                       .Eq = Eq_device_.data(),
                                       .qsum = qsum_.data(),
                                       .qcount = qcount_.data(),
                                       .nn = nn_,
                                       .ne = ne_,
                                       .nsq = nsq_,
                                       .n_workers = n_workers_};
    return device_ptrs;
  }

  void save_embedding_result(cuda_stream& destride_stream, cuda_stream& copy_stream) {
    static const size_t block_size = 128;
    static const size_t block_count = (Y_.size() + block_size - 1) / block_size;
    destride_embedding<real_t><<<block_count, block_size, 0, destride_stream.stream()>>>(Y_.data(), Y_destride_.data(), Y_.size(), nn_);
    destride_stream.sync();

    Y_destride_.get_array_async(Y_host_.data(), copy_stream.stream());
  }

  void update_frames(std::shared_ptr<sce_results<real_t>> results,
                     cuda_stream& kernel_stream, cuda_stream& copy_stream,
                     uint64_t& curr_iter, real_t& curr_Eq,
                     uint64_t next_iter, real_t next_Eq) {
    // Save the previous frame
    if (results->n_frames() > 0) {
      copy_stream.sync();
      results->add_frame(curr_iter, curr_Eq, Y_host_);
    }
    // Start copying this frame
    curr_Eq = next_Eq;
    curr_iter = next_iter;
    save_embedding_result(kernel_stream, copy_stream);
  }

  // delete move and copy to avoid accidentally using them
  sce_gpu(const sce_gpu &) = delete;
  sce_gpu(sce_gpu &&) = delete;

  int n_workers_;
  uint64_t nn_;
  uint64_t ne_;
  real_t nsq_;

  // CUDA types to run callback (Eq + progress) in the graph
  cudaHostFn_t progress_callback_fn_;
  callBackData_t<real_t> progress_callback_params_;

  // Uniform draw tables
  device_array<uint32_t> rng_state_;
  discrete_table_device<real_t> node_table_;
  discrete_table_device<real_t> edge_table_;

  // Embedding
  device_array<real_t> Y_;
  device_array<real_t> Y_destride_;
  std::vector<real_t> Y_host_;
  // Sparse distance indexes
  device_array<uint64_t> I_;
  device_array<uint64_t> J_;

  // Algorithm progress
  real_t Eq_host_;
  device_value<real_t> Eq_device_;
  device_array<real_t> qsum_;
  real_t qsum_total_host_;
  device_value<real_t> qsum_total_device_;
  device_array<uint64_t> qcount_;
  uint64_t qcount_total_host_;
  device_value<uint64_t> qcount_total_device_;

  // cub space
  size_t qsum_tmp_storage_bytes_;
  size_t qcount_tmp_storage_bytes_;
  device_array<void> qsum_tmp_storage_;
  device_array<void> qcount_tmp_storage_;
};

// These two templates are explicitly instantiated here as the instantiation
// in python_bindings.cpp is not seen by nvcc, leading to a unlinked function
// when imported
template std::shared_ptr<sce_results<float>>
wtsne_gpu<float>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                 std::vector<float> &, std::vector<float> &, const float,
                 const uint64_t, const int, const int, const uint64_t,
                 const float, const bool, const bool, const int, const int,
                 const unsigned int);
template std::shared_ptr<sce_results<double>>
wtsne_gpu<double>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                  std::vector<double> &, std::vector<double> &, const double,
                  const uint64_t, const int, const int, const uint64_t,
                  const double, const bool, const bool, const int, const int,
                  const unsigned int);

/****************************
 * Main control function    *
 ****************************/
template <typename real_t>
std::shared_ptr<sce_results<real_t>>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &dists, std::vector<real_t> &weights,
          const real_t perplexity, const uint64_t maxIter, const int block_size,
          const int n_workers, const uint64_t nRepuSamp, const real_t eta0,
          const bool bInit, const bool animated, const int cpu_threads, const int device_id,
          const unsigned int seed) {
  // Check input
  std::vector<real_t> Y;
  std::vector<double> P;
  std::tie(Y, P) =
      wtsne_init<real_t>(I, J, dists, weights, perplexity, cpu_threads, seed);

  // These classes set up and manage all of the memory
  auto results = std::make_shared<sce_results<real_t>>(animated, n_workers, maxIter);
  sce_gpu<real_t> embedding(Y, I, J, P, weights, n_workers, device_id, seed);

  // Run the algorithm
  embedding.run_SCE(results, maxIter, block_size, n_workers, nRepuSamp, eta0, bInit);

  // Get the result back
  results->add_result(maxIter, embedding.current_Eq(), embedding.get_embedding_result());
  return results;
}
