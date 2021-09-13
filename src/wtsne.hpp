/*
 ============================================================================
 Author      : Zhirong Yang
 Copyright   : Copyright by Zhirong Yang. All rights are reserved.
 Description : Stochastic Optimization for t-SNE
 ============================================================================
 */

// Modified by John Lees
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef> // size_t
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdio.h>
#include <string.h>
#include <tuple>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "uniform_discrete.hpp"

#ifndef DIM
#define DIM 2
#endif

const int n_steps = 100;
const double PERPLEXITY_TOLERANCE = 1e-5;

// Get indices where each row starts in the sparse matrix
// NB this won't work if any rows are missing
inline std::vector<uint64_t> row_start_indices(const std::vector<uint64_t> &I,
                                               const size_t n_samples) {
  std::vector<uint64_t> row_start_idx(n_samples + 1);
  size_t i_idx = 0;
  row_start_idx[0] = 0;
  row_start_idx[n_samples] = I.size();
  for (long i = 1; i < n_samples; ++i) {
    while (I[i_idx] < i) {
      i_idx++;
    }
    row_start_idx[i] = i_idx;
  }
  return row_start_idx;
}

template <typename real_t>
std::vector<double> conditional_probabilities(const std::vector<uint64_t> &I,
                                              const std::vector<uint64_t> &J,
                                              const std::vector<real_t> &dists,
                                              const uint64_t n_samples,
                                              const real_t perplexity,
                                              const int n_threads) {
  using namespace std::literals;
  const auto start = std::chrono::steady_clock::now();
  std::vector<double> P(
      dists.size()); // note double (as in sklearn implementation)
  // Simple
  if (perplexity <= 0) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (uint64_t idx = 0; idx < dists.size(); ++idx) {
      P[idx] = 1 - dists[idx];
    }
  } else {
    const std::vector<uint64_t> row_start_idx = row_start_indices(I, n_samples);
    const real_t desired_entropy = std::log(perplexity);

// Conditional Gaussians
// see _binary_search_perplexity in
// https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/_utils.pyx
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (uint64_t sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
      real_t beta_min = -std::numeric_limits<real_t>::infinity();
      real_t beta_max = std::numeric_limits<real_t>::infinity();
      real_t beta = 1.0;
      real_t sum_Pi, sum_disti_Pi, entropy, entropy_diff;
      for (int l = 0; l < n_steps; ++l) {
        sum_Pi = 0.0;
        for (uint64_t j = row_start_idx[sample_idx];
             j < row_start_idx[sample_idx + 1]; ++j) {
          P[j] = std::exp(-dists[j] * beta);
          sum_Pi += P[j];
        }

        if (sum_Pi == 0.0) {
          sum_Pi = std::numeric_limits<real_t>::epsilon();
        }
        sum_disti_Pi = 0.0;

        for (uint64_t j = row_start_idx[sample_idx];
             j < row_start_idx[sample_idx + 1]; ++j) {
          P[j] /= sum_Pi;
          sum_disti_Pi += dists[j] * P[j];
        }

        entropy = std::log(sum_Pi) + beta * sum_disti_Pi;
        entropy_diff = entropy - desired_entropy;

        if (std::abs(entropy_diff) <= PERPLEXITY_TOLERANCE) {
          break;
        }

        if (entropy_diff > 0.0) {
          beta_min = beta;
          if (beta_max == std::numeric_limits<real_t>::infinity()) {
            beta *= 2.0;
          } else {
            beta = (beta + beta_max) * 0.5;
          }
        } else {
          beta_max = beta;
          if (beta_min == -std::numeric_limits<real_t>::infinity()) {
            beta *= 0.5;
          } else {
            beta = (beta + beta_min) * 0.5;
          }
        }
      }
    }
  }
  const auto end = std::chrono::steady_clock::now();
  std::cout << "Preprocessing " << n_samples
            << " samples with perplexity = " << perplexity << " took "
            << (end - start) / 1ms << "ms" << std::endl;
  return P;
}

template <class real_t>
std::tuple<std::vector<real_t>, std::vector<double>>
wtsne_init(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
           std::vector<real_t> &dists, std::vector<double> &weights,
           const real_t perplexity, const int n_threads,
           const unsigned int seed) {
  // Check input
  if (I.size() != J.size() || I.size() != dists.size() ||
      J.size() != dists.size()) {
    throw std::runtime_error("Mismatching sizes in input vectors");
  }
  if (I.size() < 2) {
    throw std::runtime_error("Input size too small");
  }
  const uint64_t nn = weights.size();
  const uint64_t ne = dists.size();

  // Preprocess distances
  std::vector<double> P =
      conditional_probabilities<real_t>(I, J, dists, nn, perplexity, n_threads);

  // Normalise distances and weights
  normalise_vector(P, true, n_threads);
  normalise_vector(weights, true, n_threads);

  // Set starting Y0
  pRNG<real_t> rng_state(n_threads, std::vector<uint32_t>(1, seed));
  rng_state.long_jump(); // Independent RNG from SCE algorithm
  std::vector<real_t> Y(nn * DIM);
  #pragma omp parallel for schedule(static) num_threads(n_threads)
  for (int coor = 0; coor < nn * DIM; ++coor) {
#ifdef _OPENMP
    const int thread_idx = omp_get_thread_num();
#else
    static const int thread_idx = 1;
#endif
    rng_state_t<real_t> thread_rng_state = rng_state.state(thread_idx);
    Y[coor] = unif_rand(thread_rng_state) * static_cast<real_t>(1e-4);
  }

  return std::make_tuple(Y, P);
}

template <typename real_t>
inline void update_progress(const long long iter, const uint64_t maxIter,
                            const real_t eta, const real_t Eq) {
  if (iter % MAX(1, maxIter / 1000) == 0 || iter == maxIter - 1) {
    // Check for keyboard interrupt from python
    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
    fprintf(stderr, "%cOptimizing\t eta=%f Progress: %.1lf%%, Eq=%.20f", 13,
            eta, (real_t)iter / maxIter * 100, Eq);
    fflush(stderr);
  }
}

// Function prototypes
// in wtsne_cpu.cpp
std::vector<double> wtsne(const std::vector<uint64_t> &I,
                          const std::vector<uint64_t> &J,
                          std::vector<double> &dists,
                          std::vector<double> &weights, const double perplexity,
                          const uint64_t maxIter, const uint64_t nRepuSamp,
                          const double eta0, const bool bInit, const int n_workers,
                          const int n_threads, const unsigned int seed);
// in wtsne_gpu.cu
template <typename real_t>
std::vector<real_t>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &dists, std::vector<double> &weights,
          const real_t perplexity, const uint64_t maxIter, const int block_size,
          const int n_workers, const uint64_t nRepuSamp, const real_t eta0,
          const bool bInit, const int n_threads, const int device_id,
          const unsigned int seed);
