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
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#include "containers.cuh"
#include "uniform_discrete.hpp"
#include "wtsne.hpp"

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

template <typename real_t> class SCEDeviceMemory {
public:
  SCEDeviceMemory(const std::vector<real_t> &Y, const std::vector<uint64_t> &I,
                  const std::vector<uint64_t> &J, const std::vector<double> &P,
                  const std::vector<real_t> &weights, int n_workers,
                  const unsigned int seed)
      : n_workers_(n_workers), nn_(weights.size()),
        ne_(P.size()), nsq_(static_cast<real_t>(nn_) * (nn_ - 1)),
        rng_state_(load_rng<real_t>(n_workers, seed)), Y_(Y), I_(I),
        J_(J), Eq_(1.0), qsum_(n_workers), qsum_total_(0.0),
        qcount_(n_workers), qcount_total_(0) {
    // Initialise tmp space for reductions on qsum and qcount
    cub::DeviceReduce::Sum(qsum_tmp_storage_.data(), qsum_tmp_storage_bytes_,
                           qsum_.data(), qsum_total_.data(), qsum_.size());
    qsum_tmp_storage_.set_size(qsum_tmp_storage_bytes_);
    cub::DeviceReduce::Sum(qcount_tmp_storage_.data(),
                           qcount_tmp_storage_bytes_, qcount_.data(),
                           qcount_total_.data(), qcount_.size());
    qcount_tmp_storage_.set_size(qcount_tmp_storage_bytes_);

    // Set up discrete RNG tables
    node_table_ = set_device_table(weights);
    edge_table_ = set_device_table(P);
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
                                       .Eq = Eq_.data(),
                                       .qsum = qsum_.data(),
                                       .qcount = qcount_.data(),
                                       .nn = nn_,
                                       .ne = ne_,
                                       .nsq = nsq_,
                                       .n_workers = n_workers_};
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
  discrete_table_device<real_t> set_device_table(const std::vector<T>& probs) {
    discrete_table<real_t, T> table(probs);
    discrete_table_device<real_t> dev_table = { .F = table.F_table(),
                                         .A = table.A_table() };
    return dev_table;
  }

  // delete move and copy to avoid accidentally using them
  SCEDeviceMemory(const SCEDeviceMemory &) = delete;
  SCEDeviceMemory(SCEDeviceMemory &&) = delete;

  int n_workers_;
  uint64_t nn_;
  uint64_t ne_;
  real_t nsq_;

  // Uniform draw tables
  device_array<uint32_t> rng_state_;
  discrete_table_device<real_t> node_table_;
  discrete_table_device<real_t> edge_table_;

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
};

/****************************
 * Kernels                  *
 ****************************/

// Updates the embedding
template <typename real_t>
KERNEL void wtsneUpdateYKernel(
    uint32_t * rng_state, const discrete_table_ptrs<real_t> node_table,
    const discrete_table_ptrs<real_t> edge_table, volatile real_t *Y, uint64_t *I,
    uint64_t *J, real_t *Eq, real_t *qsum, uint64_t *qcount, uint64_t nn,
    uint64_t ne, real_t eta, uint64_t nRepuSamp, real_t nsq, real_t attrCoef,
    int n_workers) {
  // Worker index based on CUDA launch parameters
  int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const real_t one = 1.0; // Used a few times, fp64/fp32
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
          // But try again if another worker has written to the same location
          if (atomicAdd((real_t*)Y + d + lk, gain) != Yk_read[d] ||
              atomicAdd((real_t*)Y + d + ll, -gain) != Yl_read[d]) {
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
          __threadfence();
          r--;
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
 * Main control function     *
 ****************************/
// These two templates are explicitly instantiated here as the instantiation
// in python_bindings.cpp is not seen by nvcc, leading to a unlinked function
// when imported
template std::vector<float>
wtsne_gpu<float>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                 std::vector<float> &, std::vector<float> &, const float,
                 const uint64_t, const int, const int, const uint64_t,
                 const float, const bool, const int, const int, const unsigned int);
template std::vector<double>
wtsne_gpu<double>(const std::vector<uint64_t> &, const std::vector<uint64_t> &,
                  std::vector<double> &, std::vector<double> &, const double,
                  const uint64_t, const int, const int, const uint64_t,
                  const double, const bool, const int, const int, const unsigned int);

template <typename real_t>
std::vector<real_t>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &dists, std::vector<real_t> &weights,
          const real_t perplexity, const uint64_t maxIter, const int block_size,
          const int n_workers, const uint64_t nRepuSamp, const real_t eta0,
          const bool bInit, const int cpu_threads, const int device_id,
          const unsigned int seed) {
  // Check input
  std::vector<real_t> Y;
  std::vector<double> P;
  std::tie(Y, P) =
      wtsne_init<real_t>(I, J, dists, weights, perplexity, cpu_threads, seed);

  // Initialise CUDA
  CUDA_CALL(cudaSetDevice(device_id));

  // This class sets up and manages all of the memory
  SCEDeviceMemory<real_t> embedding(Y, I, J, P, weights, n_workers, seed);
  kernel_ptrs<real_t> device_ptrs = embedding.get_device_ptrs();

  // Main SCE loop
  const size_t block_count = (n_workers + block_size - 1) / block_size;
  for (uint64_t iter = 0; iter < maxIter; iter++) {
    real_t eta = eta0 * (1 - static_cast<real_t>(iter) / (maxIter - 1));
    eta = MAX(eta, eta0 * 1e-4);

    real_t attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    wtsneUpdateYKernel<real_t><<<block_count, block_size>>>(
        device_ptrs.rng, embedding.get_node_table(),
        embedding.get_edge_table(), device_ptrs.Y, device_ptrs.I, device_ptrs.J,
        device_ptrs.Eq, device_ptrs.qsum, device_ptrs.qcount, device_ptrs.nn,
        device_ptrs.ne, eta, nRepuSamp, device_ptrs.nsq, attrCoef, device_ptrs.n_workers);
    CUDA_CALL(cudaDeviceSynchronize());

    // Print progress
    real_t Eq = embedding.update_Eq();
    update_progress(iter, maxIter, eta, Eq);
  }
  std::cerr << std::endl << "Optimizing done" << std::endl;

  return embedding.get_embedding_result();
}
