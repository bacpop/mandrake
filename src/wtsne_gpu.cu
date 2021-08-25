/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Weighted t-SNE with stochastic optimization
 ============================================================================
 */

// Modified by John Lees 2021

#include <cfloat>
#include <curand_kernel.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#include "containers.cuh"
#include "wtsne.hpp"

/****************************
 * Classes                  *
 ****************************/
template <typedef real_t> struct gsl_table_device {
  real_t K;
  real_t *F;
  size_t *A;
};

template <typedef real_t> struct gsl_table_host {
  device_array<real_t> F;
  device_array<size_t> A;
};

template <typedef real_t> struct kernel_ptrs {
  real_t *Y;
  uint64_t *I;
  uint64_t *J;
  real_t *Eq;
  real_t *qsum;
  int *qcount;
  uint64_t nn;
  uint64_t ne;
  real_t nsq;
};

template <typename real_t>
class SCEDeviceMemory {
public:
  SCEDeviceMemory(const std::vector<real_t>& Y,
                  const std::vector<uint64_t>& I,
                  const std::vector<uint64_t>& J,
                  const std::vector<real_t>& P,
                  const std::vector<real_t>& weights,
                  int block_size,
                  int block_count) :
    n_workers_(block_size * block_count),
    nn_(weights.size()),
    ne_(P.size()),
    nsq_(static_cast<real_t>(nn) * (nn - 1)),
    Y_(Y),
    I_(I),
    J_(J),
    qsum_(n_workers_),
    qcount_(n_workers_) {
      // Initialise Eq, which is managed memory between host and device
      CUDA_CALL(cudaMallocManaged((void**)&Eq_, sizeof(real_t)));
      CUDA_CALL(cudaMallocManaged((void**)&qsum_total_, sizeof(real_t)));
      CUDA_CALL(cudaMallocManaged((void**)&qcount_total_, sizeof(uint64_t)));
      *Eq_ = 1;
      *qsum_total_ = 0;
      *qcount_total_ = 0;

      // Initialise tmp space for reductions on qsum and qcount
      cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                             qsum_.data(), qsum_total_.data(), qsum_.size());
      qsum_tmp_storage_.set_size(qsum_tmp_storage_bytes_);
      cub::DeviceReduce::Sum(qcount_tmp_storage_.data(), qcount_tmp_storage_bytes_,
                             qcount_.data(), qcount_total_.data(), qcount_.size());
      qcount_tmp_storage_.set_size(qcount_tmp_storage_bytes_);

      // Set up discrete RNG tables
      gsl_rng_env_setup();
      node_table_ = set_device_table(weights);
      edge_table_ = set_device_table(P);
  }

  ~SCEDeviceMemory() {
    CUDA_CALL(cudaFree((void *)Eq_));
    CUDA_CALL(cudaFree((void *)qsum_total_));
    CUDA_CALL(cudaFree((void *)qcount_total_));
  }

  gsl_table_device<real_t> get_node_table() {
    gsl_table_device<real_t> device_node_table =
                     { .K = node_table_.F.size(),
                       .F = node_table_.F.data(),
                       .A = node_table_.A.data() };
    return device_node_table;
  }

  gsl_table_device<real_t> get_edge_table() {
    gsl_table_device<real_t> device_edge_table =
                     { .K = edge_table_.F.size(),
                       .F = edge_table_.F.data(),
                       .A = edge_table_.A.data() };
    return device_edge_table;
  }

  kernel_ptrs<real_t> get_device_ptrs() {
    kernel_ptrs<real_T> device_ptrs =
      { .Y = Y_.data(),
        .I = I_.data(),
        .J = J_.data(),
        .Eq = Eq_,
        .qsum = qsum_.data(),
        .qcount = qcount_.data(),
        .nn = nn_,
        .ne = ne_,
        .nsq = nsq_ };
    return device_ptrs;
  }

  std::vector<real_t> get_embedding_result() {
    std::vector<real_t> Y_host(Y_.size());
    Y_.get_array(Y_host);
    return Y_host;
  }

  real_t update_Eq() {
    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
      qsum_.data(), qsum_total_, qsum_.size());
    cub::DeviceReduce::Sum(qcount_tmp_storage_.data(), qcount_tmp_storage_bytes_,
      qcount_.data(), qcount_total_, qcount_.size());
    CUDA_CALL(cudaDeviceSynchronize());

    real_t Eq = ((*Eq_) * nsq_ + (*d_qsum_total_)) / (nsq_ + (*d_qcount_total_));
    *Eq_ = Eq;
    return Eq;
  }

private:
  template <typename real_t, typename T>
  gsl_table_host<real_t> set_device_table(const std::vector<T>& weights) {
    uint64_t table_size = weights.size();
    gsl_ran_discrete_t *gsl_table = gsl_ran_discrete_preproc(table_size,
                                                             weights.data());
    gsl_table_host<real_t> device_table;
    device_table.F = device_array<real_t>(table_size);
    device_table.F.set_array(gsl_table->F);
    device_table.A = device_array<size_t>(table_size);
    device_table.A.set_array(gsl_table->A);

    gsl_ran_discrete_free(gsl_table);
    return device_table;
  }

  // delete move and copy to avoid accidentally using them
  SCEDeviceMemory ( const SCEDeviceMemory & ) = delete;
  SCEDeviceMemory ( SCEDeviceMemory && ) = delete;

  int n_workers_;
  real_t nsq_;
  uint64_t nn_;
  uint64_t ne_;

  // Uniform draw tables
  gsl_table_host node_table_;
  gsl_table_host edge_table_;

  // Embedding
  device_array<real_t> Y_;
  // Sparse distance indexes
  device_array<uint64_t> I_;
  device_array<uint64_t> J_;

  // Algorithm progress
  volatile real_t *Eq_;
  device_array<real_t> qsum_;
  volatile real_t *qsum_total_;
  device_array<uint64_t> qcount_;
  volatile uint64_t *qcount_total_;

  // cub space
  size_t qsum_tmp_storage_bytes_;
  size_t qcount_tmp_storage_bytes_;
  device_array<void> qsum_tmp_storage_;
  device_array<void> qcount_tmp_storage_;
};

/****************************
 * Device functions         *
 ****************************/

template <typedef real_t>
__device__ size_t discrete_draw(curandState *state,
                                const gsl_table_device<real_t> &unif_table) {
  size_t c = 0;
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

__global__ void setup_rng_kernel(curandState *state, const long n_draws, int seed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws;
       i += blockDim.x * gridDim.x) {
    curand_init(seed, i, 0, &state[i]);
  }
}

// Updates the embedding
template <typedef real_t>
__global__ void
wtsneUpdateYKernel(curandState *rng_state,
                   const gsl_table_device<real_t> node_table,
                   const gsl_table_device<real_t> edge_table,
                   real_t *Y,
                   uint64_t *I,
                   uint64_t *J,
                   real_t *Eq,
                   real_t *qsum,
                   int *qcount,
                   uint64_t nn,
                   uint64_t ne,
                   real_t eta,
                   uint64_t nRepuSamp,
                   real_t nsq,
                   real_t attrCoef) {
  // Bring RNG state into local registers
  int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = rng_state[workerIdx];

  real_t dY[DIM];
  real_t Yk_read[DIM];
  real_t Yl_read[DIM];
  real_t c = static_cast<real_t>(1.0) / ((*Eq) * nsq);
  qsum[workerIdx] = static_cast<real_t>(0.0);
  qcount[workerIdx] = 0;

  real_t repuCoef = 2 * c / nRepuSamp * nsq;
  for (int r = 0; r < nRepuSamp + 1; r++) {
    uint64_t k, l;
    if (r == 0) {
      uint64_t e =
          static_cast<uint64_t>(discrete_draw(localState, edge_table) % ne);
      k = I[e];
      l = J[e];
    } else {
      k = static_cast<uint64_t>(discrete_draw(localState, node_table) % nn);
      l = static_cast<uint64_t>(discrete_draw(localState, node_table) % nn);
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
        if (atomicCAS(Y + d + lk, Yk_read[d], Yk_read[d] + gain) != Yk_read[d] ||
            atomicCAS(Y + d + ll, Yl_read[d], Yl_read[d] - gain) != Yl_read[d]) {
          overwrite = true;
        }
      }
      if (!overwrite) {
        qsum[workerIdx] += q;
        qcount[workerIdx]++;
      }
    }
    __syncwarp();
  }
  // Store RNG state back to global
  state[i] = localState;
}

/****************************
 * Main control function     *
 ****************************/
template <typename real_t>
std::vector<real_t> wtsne_gpu(std::vector<uint64_t> &I,
                             std::vector<uint64_t> &J,
                             std::vector<real_t> &P,
                             std::vector<real_t> &weights,
                             uint64_t maxIter,
                             int block_size,
                             int block_count,
                             uint64_t nRepuSamp,
                             real_t eta0,
                             bool bInit) {
  // Check input
  std::vector<real_t> Y = wtsne_init<real_t>(I, J, P, weights);

  // Initialise CUDA
  CUDA_CALL(cudaSetDevice(0));

  // This class sets up and manages all of the memory
  SCEDeviceMemory embedding<real_t>(Y, I, J, P, weights, block_size, block_count);
  kernel_ptrs device_ptrs<real_t> = embedding.get_device_ptrs();

  // Set up random number generation for device
  const int n_workers = block_size * block_count;
  const size_t rng_block_count = (n_workers + block_size - 1) / block_size;
  curandState *device_rng;
  setup_rng_kernel<<<rng_block_count, block_size>>>(device_rng, n_workers);
  CUDA_CALL(cudaDeviceSynchronize());

  // Main SCE loop
  for (uint64_t iter = 0; iter < maxIter; iter++) {
    real_t eta = eta0 * (1 - static_cast<real_t>(iter) / (maxIter - 1));
    eta = MAX(eta, eta0 * 1e-4);

    real_t attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    wtsneUpdateYKernel<real_t><<<block_count, block_size>>>(
      device_rng,
      embedding.get_node_table(),
      embedding.get_edge_table(),
      device_ptrs.Y,
      device_ptrs.I,
      device_ptrs.J,
      device_ptrs.Eq,
      device_ptrs.qsum,
      device_ptrs.qcount,
      device_ptrs.nn,
      device_ptrs.ne,
      eta,
      nRepuSamp,
      device_ptrs.nsq,
      attrCoef);
    CUDA_CALL(cudaDeviceSynchronize());
    real_t Eq = embedding.update_Eq();

    // Print progress
    if (iter % MAX(1, maxIter / 1000) == 0 || iter == maxIter - 1) {
      fprintf(stderr, "%cOptimizing (GPU)\t eta=%f Progress: %.1lf%%, Eq=%.20f", 13, eta, (real_t)iter / maxIter * 100, Eq);
      fflush(stderr);
    }
  }
  std::cerr << std::endl << "Optimizing done" << std::endl;

  std::vector<real_t> embedding_result = embedding.get_embedding_result();
  return embedding_result;
}
