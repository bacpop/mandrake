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

#include "containers.cuh"
#include "wtsne.hpp"

template <typedef real_t> struct gsl_table_device {
  real_t K;
  real_t *F;
  size_t *A;
};

template <typedef real_t> struct gsl_table_host {
  device_array<real_t> F;
  device_array<size_t> *A;
};

template <typename real_t>
class SCEDeviceMemory {
public:
  SCEDeviceMemory(const std::vector<real_t>& Y,
                  const std::vector<size_t>& I,
                  const std::vector<size_t>& J,
                  const std::vector<real_t>& P,
                  int block_size,
                  int block_count) :
    n_workers_(block_size * block_count),
    Y_(Y), I_(I), J_(J) {

  }

  ~SCEDeviceMemory();

  gsl_table_device<real_t> get_node_table {
    gsl_table_device<real_t> device_node_table = 
                     { .K = node_table_.F.size(),
                       .F = node_table_.F.data(),
                       .A = node_table_.A.data() };
    return device_node_table;
  }

  gsl_table_device<real_t> get_edge_table {
    gsl_table_device<real_t> device_edge_table = 
                     { .K = edge_table_.F.size(),
                       .F = edge_table_.F.data(),
                       .A = edge_table_.A.data() };
    return device_edge_table;
  }

private:
  // delete move and copy to avoid accidentally using them
  SCEDeviceMemory ( const SCEDeviceMemory & ) = delete;
  SCEDeviceMemory ( SCEDeviceMemory && ) = delete;

  // Uniform draw tables
  gsl_table_host node_table_;
  gsl_table_host edge_table_;

  // Sparse distances
  device_array<real_t> Y_;
  device_array<uint64_t> I_;
  device_array<uint64_t> J_;

  // Algorithm progress
  int n_workers_;
  real_t *Eq_;
  device_array<real_t> qsum_;
  device_array<real_t> qsum_total_;
  device_array<int> qcount_;
  device_array<int> qcount_total_;

  // cub space
  size_t tmp_storage_bytes_;
  device_array<void> tmp_storage_;
};

/****************************
 * Functions run on the      *
 * device                    *
 ****************************/
template <typedef real_t>
__device__ size_t discrete_draw(curandState *state,
                                const gsl_table<real_t> &unif_table) {
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

__global__ void setup_kernel(curandState *state, const long n_draws) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws;
       i += blockDim.x * gridDim.x) {
    /* Each thread gets same seed, a different sequence
        number, no offset */
    curand_init(314159, i, 0, &state[i]);
  }
}

// TODO: this can just be a memcpy in the gsl class
__global__ void assembleGSLKernel(gsl_ran_discrete_t *d_gsl_de,
                                  size_t *d_gsl_de_A, double *d_gsl_de_F,
                                  gsl_ran_discrete_t *d_gsl_dn,
                                  size_t *d_gsl_dn_A, double *d_gsl_dn_F) {
  d_gsl_de->A = d_gsl_de_A;
  d_gsl_de->F = d_gsl_de_F;
  d_gsl_dn->A = d_gsl_dn_A;
  d_gsl_dn->F = d_gsl_dn_F;
}

// Updates the embedding
template <typedef real_t>
__global__ void
wtsneUpdateYKernel(curandState *rng_state,
                   const gsl_table<real_t> *node_table,
                   const gsl_table<real_t> *edge_table,
                   real_t *Y, uint64_t *I, uint64_t *J, real_t *d_Eq,
                   real_t *qsum, int *qcount, uint64_t nn, uint64_t ne,
                   real_t eta, int nRepuSamp, real_t nsq,
                   real_t attrCoef) {
  // Bring RNG state into local registers
  int workerIdx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = rng_state[workerIdx];

  real_t dY[DIM];
  real_t Yk_read[DIM];
  real_t Yl_read[DIM];
  real_t c = static_cast<real_t>(1.0) / ((*d_Eq) * nsq);
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
      k = static_cast<uint64_t>(my_curand_discrete(localState, node_table) % nn);
      l = static_cast<uint64_t>(my_curand_discrete(localState, node_table) % nn);
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

__global__ void resetQsumQCountTotalKernel(float *d_qsum_total,
                                           int *d_qcount_total) {
  (*d_qsum_total) = 0.0;
  (*d_qcount_total) = 0;
}

template <typename T>
__global__ void reduceSumArrayKernel(T *array, int n, T *arraySum) {
  T sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    sum += array[i];
  }
  atomicAdd(arraySum, sum);
}

__global__ void updateEqKernel(float *d_Eq, float *d_qsum_total,
                               int *d_qcount_total, float nsq) {
  (*d_Eq) = ((*d_Eq) * nsq + (*d_qsum_total)) / (nsq + (*d_qcount_total));
}

/****************************
 * Functions to move data    *
 * and functions on/off GPUs *
 ****************************/

// Moves arrays onto GPU
void allocateDataAndCopy2Device(float *&d_Eq, long long nn, long long ne,
                                float *&d_qsum, int *&d_qcount,
                                float *&d_qsum_total, int *&d_qcount_total,
                                int nWorker) {
  float Eq = 1;
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_Eq, sizeof(float)));
  CUDA_CHECK_RETURN(
      cudaMemcpy(d_Eq, &Eq, sizeof(float), cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_qsum, nWorker * sizeof(float)));
  CUDA_CHECK_RETURN(
      cudaMalloc((void **)&d_qcount, nWorker * sizeof(long long)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_qsum_total, sizeof(float)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_qcount_total, sizeof(long long)));
}

void setupDiscreteDistribution(
    curandState *&d_nnStates1, curandState *&d_nnStates2,
    curandState *&d_neStates, std::vector<double> &P,
    std::vector<double> &weights, gsl_ran_discrete_t *&d_gsl_de,
    gsl_ran_discrete_t *&d_gsl_dn, double *&d_gsl_de_F, double *&d_gsl_dn_F,
    size_t *&d_gsl_de_A, size_t *&d_gsl_dn_A, int blockCount, int blockSize,
    long long nn, long long ne) {
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_nnStates1,
                               blockCount * blockSize * sizeof(curandState)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_nnStates2,
                               blockCount * blockSize * sizeof(curandState)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_neStates,
                               blockCount * blockSize * sizeof(curandState)));
  setupCURANDKernel<<<blockCount, blockSize>>>(d_nnStates1, d_nnStates2,
                                               d_neStates);

  // These are free'd at the end of the function
  gsl_rng_env_setup();
  gsl_ran_discrete_t *gsl_de = gsl_ran_discrete_preproc(ne, P.data());
  gsl_ran_discrete_t *gsl_dn = gsl_ran_discrete_preproc(nn, weights.data());

  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_de, sizeof(gsl_ran_discrete_t)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_dn, sizeof(gsl_ran_discrete_t)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_de_A, sizeof(size_t) * ne));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_de_F, sizeof(double) * ne));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_dn_A, sizeof(size_t) * nn));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&d_gsl_dn_F, sizeof(double) * nn));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_de, gsl_de, sizeof(gsl_ran_discrete_t),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_de_A, gsl_de->A, sizeof(size_t) * ne,
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_de_F, gsl_de->F, sizeof(double) * ne,
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_dn, gsl_dn, sizeof(gsl_ran_discrete_t),
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_dn_A, gsl_dn->A, sizeof(size_t) * nn,
                               cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(d_gsl_dn_F, gsl_dn->F, sizeof(double) * nn,
                               cudaMemcpyHostToDevice));
  assembleGSLKernel<<<1, 1>>>(d_gsl_de, d_gsl_de_A, d_gsl_de_F, d_gsl_dn,
                              d_gsl_dn_A, d_gsl_dn_F);
  gsl_ran_discrete_free(gsl_de);
  gsl_ran_discrete_free(gsl_dn);
}

// Frees memory on GPU
void freeDataInDevice(float *&d_qsum, int *&d_qcount, float *&d_qsum_total,
                      int *&d_qcount_total, curandState *&d_nnStates1,
                      curandState *&d_nnStates2, curandState *&d_neStates,
                      gsl_ran_discrete_t *&d_gsl_de,
                      gsl_ran_discrete_t *&d_gsl_dn, double *&d_gsl_de_F,
                      double *&d_gsl_dn_F, size_t *&d_gsl_de_A,
                      size_t *&d_gsl_dn_A) {
  // data
  CUDA_CHECK_RETURN(cudaFree(d_qsum));
  CUDA_CHECK_RETURN(cudaFree(d_qcount));
  CUDA_CHECK_RETURN(cudaFree(d_qsum_total));
  CUDA_CHECK_RETURN(cudaFree(d_qcount_total));

  // rng
  CUDA_CHECK_RETURN(cudaFree(d_nnStates1));
  CUDA_CHECK_RETURN(cudaFree(d_nnStates2));
  CUDA_CHECK_RETURN(cudaFree(d_neStates));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_de_A));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_de_F));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_de));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_dn_A));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_dn_F));
  CUDA_CHECK_RETURN(cudaFree(d_gsl_dn));
}

/****************************
 * Main control function     *
 ****************************/
std::vector<float> wtsne_gpu(std::vector<long long> &I,
                             std::vector<long long> &J, std::vector<double> &P,
                             std::vector<double> &weights, long long maxIter,
                             int blockSize, int blockCount, long long nRepuSamp,
                             double eta0, bool bInit) {
  // Check input
  std::vector<float> Y = wtsne_init<float>(I, J, P, weights);
  long long nn = weights.size();
  long long ne = P.size();

  // Initialise CUDA
  cudaSetDevice(0);
  cudaDeviceReset();
  int nWorker = blockSize * blockCount;
  float nsq = (float)nn * (nn - 1);

  // Create pointers for mallocs
  float *d_Eq;
  float *d_qsum, *d_qsum_total;
  int *d_qcount, *d_qcount_total;

  // malloc on device
  allocateDataAndCopy2Device(d_Eq, nn, ne, d_qsum, d_qcount, d_qsum_total,
                             d_qcount_total, nWorker);
  thrust::device_vector<float> d_Y = Y;
  thrust::device_vector<long long> d_I = I;
  thrust::device_vector<long long> d_J = J;

  // Set up random number generation
  curandState *d_nnStates1, *d_nnStates2;
  curandState *d_neStates;
  gsl_ran_discrete_t *d_gsl_de = nullptr;
  gsl_ran_discrete_t *d_gsl_dn = nullptr;
  double *d_gsl_de_F, *d_gsl_dn_F;
  size_t *d_gsl_de_A, *d_gsl_dn_A;
  setupDiscreteDistribution(d_nnStates1, d_nnStates2, d_neStates, P, weights,
                            d_gsl_de, d_gsl_dn, d_gsl_de_F, d_gsl_dn_F,
                            d_gsl_de_A, d_gsl_dn_A, blockCount, blockSize, nn,
                            ne);

  // Main SCE loop
  float *d_Y_array = thrust::raw_pointer_cast(&d_Y[0]);
  long long *d_I_array = thrust::raw_pointer_cast(&d_I[0]);
  long long *d_J_array = thrust::raw_pointer_cast(&d_J[0]);
  for (long long iter = 0; iter < maxIter; iter++) {
    float eta = eta0 * (1 - (float)iter / (maxIter - 1));
    eta = MAX(eta, eta0 * 1e-4);

    float attrCoef = (bInit && iter < maxIter / 10) ? 8 : 2;
    wtsneUpdateYKernel<<<blockCount, blockSize>>>(
        d_nnStates1, d_nnStates2, d_neStates, d_gsl_dn, d_gsl_de, d_Y_array,
        d_I_array, d_J_array, d_Eq, d_qsum, d_qcount, nn, ne, eta, nRepuSamp,
        nsq, attrCoef);

    resetQsumQCountTotalKernel<<<1, 1>>>(d_qsum_total, d_qcount_total);
    reduceSumArrayKernel<<<16, 128>>>(d_qsum, nWorker, d_qsum_total);
    reduceSumArrayKernel<<<16, 128>>>(d_qcount, nWorker, d_qcount_total);
    updateEqKernel<<<1, 1>>>(d_Eq, d_qsum_total, d_qcount_total, nsq);

    // Print progress
    if (iter % MAX(1, maxIter / 1000) == 0 || iter == maxIter - 1) {
      fprintf(stderr, "%cOptimizing progress (GPU): %.1lf%%", 13,
              (float)iter / maxIter * 100);
      fflush(stderr);
    }
  }
  std::cerr << std::endl << "Optimizing done" << std::endl;

  // Free memory on GPU
  freeDataInDevice(d_qsum, d_qcount, d_qsum_total, d_qcount_total, d_nnStates1,
                   d_nnStates2, d_neStates, d_gsl_de, d_gsl_dn, d_gsl_de_F,
                   d_gsl_dn_F, d_gsl_de_A, d_gsl_dn_A);

  // Get the result from device
  try {
    thrust::copy(d_Y.begin(), d_Y.end(), Y.begin());
  } catch (thrust::system_error &e) {
    // output an error message and exit
    std::cerr << "Error getting result: " << e.what() << std::endl;
    // exit(1);
  }

  return Y;
}
