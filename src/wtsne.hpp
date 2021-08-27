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

#ifndef DIM
#define DIM 2
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename real_t>
typename std::enable_if<!std::is_same<real_t, float>::value, std::vector<float>>::type
convert_vector(const std::vector<real_t>& d_vec) {
  std::vector<float> f_vec(d_vec.begin(), d_vec.end());
  return f_vec;
}

// No conversion needed
template <typename real_t>
typename std::enable_if<std::is_same<real_t, double>::value, std::vector<float>>::type
convert_vector(const std::vector<real_t>& d_vec) {
  return d_vec;
}

// Get indices where each row starts in the sparse matrix
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

const int n_steps = 100;
const double PERPLEXITY_TOLERANCE = 1e-5;

template <typename real_t>
std::vector<real_t> conditional_probabilities(const std::vector<uint64_t> &I,
                                              const std::vector<uint64_t> &J,
                                              const std::vector<real_t> &dists,
                                              const uint64_t n_samples,
                                              const real_t perplexity,
                                              const int n_threads) {
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
  return convert_vector(P);
}

template <typename T>
inline void normalise_vector(std::vector<T> &vec, const int n_threads) {
  T sum = static_cast<T>(0.0);
#pragma omp parallel for schedule(static) reduction(+: sum) num_threads(n_threads)
  for (uint64_t it = 0; it < vec.size(); ++it) {
    sum += vec[it];
  }
  sum = MAX(sum, std::numeric_limits<T>::epsilon());
#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (uint64_t it = 0; it < vec.size(); ++it) {
    vec[it] /= sum;
  }
}

template <class real_t>
std::tuple<std::vector<real_t>, std::vector<real_t>>
wtsne_init(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
           std::vector<real_t> &dists, std::vector<real_t> &weights,
           const real_t perplexity, const int n_threads, const int seed) {
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
  std::vector<real_t> P =
      conditional_probabilities<real_t>(I, J, dists, nn, perplexity, n_threads);

  // Normalise distances and weights
  normalise_vector(P, n_threads);
  normalise_vector(weights, n_threads);

  // Set starting Y0
  // Not parallelised, but could be (or easy to do in CUDA too)
  std::mt19937 mersenne_engine{seed};
  std::uniform_real_distribution<real_t> distribution(0.0, 1e-4);
  auto gen = [&distribution, &mersenne_engine]() {
    return distribution(mersenne_engine);
  };

  std::vector<real_t> Y(nn * DIM);
  std::generate(Y.begin(), Y.end(), gen);
  return std::make_tuple(Y, P);
}

std::vector<double>
wtsne(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
      std::vector<double> &dists, std::vector<double> &weights,
      const double perplexity, const uint64_t maxIter, const uint64_t nRepuSamp,
      const double eta0, const bool bInit, const int n_threads, const int seed);

template <typename real_t>
std::vector<real_t>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &dists, std::vector<real_t> &weights,
          const real_t perplexity, const uint64_t maxIter, const int block_size,
          const int block_count, const uint64_t nRepuSamp, const real_t eta0,
          const bool bInit, const int n_threads, const int device_id,
          const int seed);
