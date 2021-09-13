#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename T>
inline void normalise_vector(std::vector<T> &vec, const bool check_positive, const int n_threads) {
  T sum = static_cast<T>(0.0);
  bool all_positive = true;
#pragma omp parallel for schedule(static) reduction(+: sum) reduction(&: all_positive) num_threads(n_threads)
  for (uint64_t it = 0; it < vec.size(); ++it) {
    sum += vec[it];
    all_positive &= vec[it] >= 0;
  }

  if (check_positive && !all_positive) {
    throw std::runtime_error("Probability vector has negative entries");
  }

  sum = MAX(sum, std::numeric_limits<T>::epsilon());
#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (uint64_t it = 0; it < vec.size(); ++it) {
    vec[it] /= sum;
  }
}
