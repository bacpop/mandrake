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
#include <curand_kernel.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#include "containers.cuh"
#include "wtsne.hpp"

/****************************
 * Classes                  *
 ****************************/
template <typename real_t> struct gsl_table_device {
  size_t K;
  real_t *F;
  size_t *A;
};

template <typename real_t> struct gsl_table_host {
  device_array<real_t> F;
  device_array<size_t> A;
};

template <typename real_t> struct kernel_ptrs {
  real_t *Y;
  uint64_t *I;
  uint64_t *J;
  real_t *Eq;
  real_t *qsum;
  uint64_t *qcount;
  uint64_t nn;
  uint64_t ne;
  real_t nsq;
};

template <typename real_t> class SCEDeviceMemory {
public:
  SCEDeviceMemory(const std::vector<real_t> &Y, const std::vector<uint64_t> &I,
                  const std::vector<uint64_t> &J, const std::vector<real_t> &P,
                  const std::vector<real_t> &weights, int block_size,
                  int block_count)
      : n_workers_(block_size * block_count), nn_(weights.size()),
        ne_(P.size()), nsq_(static_cast<real_t>(nn_) * (nn_ - 1)), Y_(Y), I_(I),
        J_(J), Eq_(1.0), qsum_(n_workers_), qsum_total_(0.0),
        qcount_(n_workers_), qcount_total_(0) {
    // Initialise tmp space for reductions on qsum and qcount
    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                           qsum_.data(), qsum_total_.data(), qsum_.size());
    qsum_tmp_storage_.set_size(qsum_tmp_storage_bytes_);
    cub::DeviceReduce::Sum(qcount_tmp_storage_.data(),
                           qcount_tmp_storage_bytes_, qcount_.data(),
                           qcount_total_.data(), qcount_.size());
    qcount_tmp_storage_.set_size(qcount_tmp_storage_bytes_);

    // Set up discrete RNG tables
    gsl_rng_env_setup();
    node_table_ = set_device_table(weights);
    edge_table_ = set_device_table(P);
  }

  gsl_table_device<real_t> get_node_table() {
    gsl_table_device<real_t> device_node_table = {.K = node_table_.F.size(),
                                                  .F = node_table_.F.data(),
                                                  .A = node_table_.A.data()};
    return device_node_table;
  }

  gsl_table_device<real_t> get_edge_table() {
    gsl_table_device<real_t> device_edge_table = {.K = edge_table_.F.size(),
                                                  .F = edge_table_.F.data(),
                                                  .A = edge_table_.A.data()};
    return device_edge_table;
  }

  kernel_ptrs<real_t> get_device_ptrs() {
    kernel_ptrs<real_t> device_ptrs = {.Y = Y_.data(),
                                       .I = I_.data(),
                                       .J = J_.data(),
                                       .Eq = Eq_.data(),
                                       .qsum = qsum_.data(),
                                       .qcount = qcount_.data(),
                                       .nn = nn_,
                                       .ne = ne_,
                                       .nsq = nsq_};
    return device_ptrs;
  }

  std::vector<real_t> get_embedding_result() {
    std::vector<real_t> Y_host(Y_.size());
    Y_.get_array(Y_host);
    return Y_host;
  }

  real_t update_Eq() {
    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                           qsum_.data(), qsum_total_.data(), qsum_.size());
    cub::DeviceReduce::Sum(qcount_tmp_storage_.data(),
                           qcount_tmp_storage_bytes_, qcount_.data(),
                           qcount_total_.data(), qcount_.size());
    CUDA_CALL(cudaDeviceSynchronize());

    real_t Eq = Eq_.get_value();
    real_t qsum_total = qsum_total_.get_value();
    uint64_t qcount_total = qcount_total_.get_value();
    Eq = (Eq * nsq_ + qsum_total) / (nsq_ + qcount_total);
    Eq_.set_value(Eq);
    return Eq;
  }

private:
  template <typename T>
      typename std::enable_if < !std::is_same<real_t, double>::value,
      gsl_table_host<real_t>::type
      set_device_table(const std::vector<T> &weights) {
    uint64_t table_size = weights.size();
    gsl_ran_discrete_t *gsl_table =
        gsl_ran_discrete_preproc(table_size, weights.data());

    // Convert type from double to real_t in F table
    std::vector<real_t> F_tab_flt(table_size);
    double *gsl_ptr = gsl_table->F;
    for (size_t i = 0; i < table_size; ++i) {
      F_tab_tlf[i] = static_cast<real_t>(*(gsl_ptr + i));
    }

    gsl_table_host<real_t> device_table;
    device_table.F = device_array<real_t>(table_size);
    device_table.F.set_array(F_tab_flt.data());
    device_table.A = device_array<size_t>(table_size);
    device_table.A.set_array(gsl_table->A);

    gsl_ran_discrete_free(gsl_table);
    return device_table;
  }

  // Double specialisation doesn't need type conversion of GSL table
  template <typename T>
      typename std::enable_if < std::is_same<real_t, double>::value,
      gsl_table_host<real_t>::type set_device_table(
          const std::vector<T> &weights) uint64_t table_size = weights.size();
  gsl_ran_discrete_t *gsl_table =
      gsl_ran_discrete_preproc(table_size, weights.data());
  gsl_table_host<double> device_table;
  device_table.F = device_array<double>(table_size);
  device_table.F.set_array(gsl_table->F);
  device_table.A = device_array<size_t>(table_size);
  device_table.A.set_array(gsl_table->A);

  gsl_ran_discrete_free(gsl_table);
  return device_table;
}

// delete move and copy to avoid accidentally using them
SCEDeviceMemory(const SCEDeviceMemory &) = delete;
SCEDeviceMemory(SCEDeviceMemory &&) = delete;

int n_workers_;
real_t nsq_;
uint64_t nn_;
uint64_t ne_;

// Uniform draw tables
gsl_table_host<real_t> node_table_;
gsl_table_host<real_t> edge_table_;

// Embedding
device_array<real_t> Y_;
// Sparse distance indexes
device_array<uint64_t> I_;
device_array<uint64_t> J_;

// Algorithm progress
device_value<real_t> Eq_;
device_array<real_t> qsum_;
device_value<real_t> qsum_total_;
device_array<uint64_t> qcount_;
device_value<uint64_t> qcount_total_;

// cub space
size_t qsum_tmp_storage_bytes_;
size_t qcount_tmp_storage_bytes_;
device_array<void> qsum_tmp_storage_;
device_array<void> qcount_tmp_storage_;
}
;

/****************************
 * Device functions         *
 ****************************/

template <typename real_t>
__device__ size_t discrete_draw(curandState *state,
                                const gsl_table_device<real_t> &unif_table) {
  real_t u = curand_uniform(state);
  size_t c = u * unif_table.K;
  real_t f = unif_table.F[c];

  real_t draw;
  if (f == 1.0 || u < f) {
    draw = c;
  } else {
    draw = unif_table.A[c];
  }
  __syncwarp();
  return draw;
}

/****************************
 * Kernels                  *
 ****************************/

__global__ void setup_rng_kernel(curandState *state, const long n_draws,
                                 int seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws;
       i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &state[i]);
  }
}

// Updates the embedding
template <typename real_t>
__global__ void wtsneUpdateYKernel(
    curandState *rng_state, const gsl_table_device<real_t> node_table,
    const gsl_table_device<real_t> edge_table, real_t *Y, uint64_t *I,
    uint64_t *J, real_t *Eq, real_t *qsum, uint64_t *qcount, uint64_t nn,
    uint64_t ne, real_t eta, uint64_t nRepuSamp, real_t nsq, real_t attrCoef) {
  // Worker index based on CUDA launch parameters
  int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
  // Bring RNG state into local registers
  curandState localState = rng_state[workerIdx];

  real_t dY[DIM];
  real_t Yk_read[DIM];
  real_t Yl_read[DIM];
  real_t c = static_cast<real_t>(1.0) / ((*Eq) * nsq);

  real_t qsum_local = 0.0;
  uint64_t qcount_local = 0;

  real_t repuCoef = 2 * c / nRepuSamp * nsq;
  for (int r = 0; r < nRepuSamp + 1; r++) {
    uint64_t k, l;
    if (r == 0) {
      uint64_t e =
          static_cast<uint64_t>(discrete_draw(&localState, edge_table) % ne);
      k = I[e];
      l = J[e];
    } else {
      k = static_cast<uint64_t>(discrete_draw(&localState, node_table) % nn);
      l = static_cast<uint64_t>(discrete_draw(&localState, node_table) % nn);
    }

    if (k != l) {
      uint64_t lk = k * DIM;
      uint64_t ll = l * DIM;
      real_t dist2 = static_cast<real_t>(0.0);
#pragma unroll
      for (int d = 0; d < DIM; d++) {
        // These are read here to avoid multiple workers writing to the same
        // location below
        Yk_read[d] = Y[d + lk];
        Yl_read[d] = Y[d + ll];
        dY[d] = Yk_read[d] - Yl_read[d];
        dist2 += dY[d] * dY[d];
      }
      real_t q = static_cast<real_t>(1.0) / (1 + dist2);

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
        // But try again if another worker has written to the same location
        if (atomicAdd(Y + d + lk, gain) != Yk_read[d] ||
            atomicAdd(Y + d + ll, -gain) != Yl_read[d]) {
          overwrite = true;
        }
      }
      if (!overwrite) {
        qsum_local += q;
        qcount_local++;
      } else {
        // Reset values
#pragma unroll
        for (int d = 0; d < DIM; d++) {
          Y[d + lk] = Yk_read[d];
          Y[d + ll] = Yl_read[d];
        }
      }
    }
    __syncwarp();
  }
  // Store local state (RNG & counts) back to global
  rng_state[workerIdx] = localState;
  qsum[workerIdx] = qsum_local;
  qcount[workerIdx] = qcount_local;
}

/****************************
 * Main control function     *
 ****************************/
// These two templates are explicitly instantiated here as the instantiation
// in python_bindings.cpp is not seen by nvcc, leading to a unlinked function
// when imported
template std::vector<float>
wtsne_gpu<float>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                 std::vector<float> &, std::vector<float> &, const float,
                 const uint64_t, const int, const int, const uint64_t,
                 const float, const bool, const int, const int, const int);
template std::vector<double>
wtsne_gpu<double>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                  std::vector<double> &, std::vector<double> &, const double,
                  const uint64_t, const int, const int, const uint64_t,
                  const double, const bool, const int, const int, const int);

template <typename real_t>
std::vector<real_t>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &dists, std::vector<real_t> &weights,
          const real_t perplexity, const uint64_t maxIter, const int block_size,
          const int block_count, const uint64_t nRepuSamp, const real_t eta0,
          const bool bInit, const int n_threads, const int device_id,
          const int seed) {
  // Check input
  std::vector<real_t> Y, P;
  std::tie(Y, P) =
      wtsne_init<real_t>(I, J, dists, weights, perplexity, n_threads, seed);

  // Initialise CUDA
  CUDA_CALL(cudaSetDevice(device_id));

  // This class sets up and manages all of the memory
  SCEDeviceMemory<real_t> embedding(Y, I, J, P, weights, block_size,
                                    block_count);
  kernel_ptrs<real_t> device_ptrs = embedding.get_device_ptrs();

  // Set up random number generation for device
  const int n_workers = block_size * block_count;
  const size_t rng_block_count = (n_workers + block_size - 1) / block_size;
  device_array<curandState> device_rng(n_workers);
  setup_rng_kernel<<<rng_block_count, block_size>>>(device_rng.data(),
                                                    n_workers, seed);
  CUDA_CALL(cudaDeviceSynchronize());

  // Main SCE loop
  for (uint64_t iter = 0; iter < maxIter; iter++) {
    real_t eta = eta0 * (1 - static_cast<real_t>(iter) / (maxIter - 1));
    eta = MAX(eta, eta0 * 1e-4);

    real_t attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    wtsneUpdateYKernel<real_t><<<block_count, block_size>>>(
        device_rng.data(), embedding.get_node_table(),
        embedding.get_edge_table(), device_ptrs.Y, device_ptrs.I, device_ptrs.J,
        device_ptrs.Eq, device_ptrs.qsum, device_ptrs.qcount, device_ptrs.nn,
        device_ptrs.ne, eta, nRepuSamp, device_ptrs.nsq, attrCoef);
    CUDA_CALL(cudaDeviceSynchronize());
    real_t Eq = embedding.update_Eq();

    // Print progress
    if (iter % MAX(1, maxIter / 1000) == 0 || iter == maxIter - 1) {
      fprintf(stderr, "%cOptimizing (GPU)\t eta=%f Progress: %.1lf%%, Eq=%.20f",
              13, eta, (real_t)iter / maxIter * 100, Eq);
      fflush(stderr);
    }
  }
  std::cerr << std::endl << "Optimizing done" << std::endl;

  std::vector<real_t> embedding_result = embedding.get_embedding_result();
  return embedding_result;
}
