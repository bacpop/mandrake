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
#include <iostream>
#include <random>
#include <stdio.h>
#include <string.h>
#include <vector>

#ifndef DIM
#define DIM 2
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <class real_t>
std::vector<real_t>
wtsne_init(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
           std::vector<real_t> &P, std::vector<real_t> &weights,
           const int n_threads, const int seed) {
  // Check input
  if (I.size() != J.size() || I.size() != P.size() || J.size() != P.size()) {
    throw std::runtime_error("Mismatching sizes in input vectors");
  }
  if (I.size() < 2) {
    throw std::runtime_error("Input size too small");
  }
  const uint64_t nn = weights.size();
  const uint64_t ne = P.size();

  // Normalise distances and weights
  real_t Psum = 0.0;
#pragma omp parallel for schedule(static) shared(Psum, P) reduction(+: Psum) num_threads(n_threads)
  for (uint64_t e = 0; e < ne; e++) {
    Psum += P[e];
  }
#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (uint64_t e = 0; e < ne; e++) {
    P[e] /= Psum;
  }

  real_t weights_sum = 0.0;
#pragma omp parallel for schedule(static) shared(weights_sum, weights) reduction(+: weights_sum) num_threads(n_threads)
  for (long long i = 0; i < nn; i++) {
    weights_sum += weights[i];
  }
#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (long long i = 0; i < nn; i++) {
    weights[i] /= weights_sum;
  }

  // Set starting Y0
  // Not parallelised, but could be
  std::mt19937 mersenne_engine{seed};
  std::uniform_real_distribution<real_t> distribution(0.0, 1e-4);
  auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

  std::vector<real_t> Y(nn * DIM);
  std::generate(Y.begin(), Y.end(), gen);
  return Y;
}

std::vector<double> wtsne(const std::vector<uint64_t> &I,
                          const std::vector<uint64_t> &J,
                          std::vector<double> &P, std::vector<double> &weights,
                          const uint64_t maxIter, const uint64_t nRepuSamp,
                          const double eta0, const bool bInit,
                          const int n_threads, const int seed);

template <typename real_t>
std::vector<real_t>
wtsne_gpu(const std::vector<uint64_t> &I, const std::vector<uint64_t> &J,
          std::vector<real_t> &P, std::vector<real_t> &weights,
          const uint64_t maxIter, const int block_size, const int block_count,
          const uint64_t nRepuSamp, const real_t eta0, const bool bInit,
          const int n_threads, const int seed);
