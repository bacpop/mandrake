#pragma once

#include <cstddef> // size_t
#include <cstdint>
#include <cstdio>

#include <pybind11/pybind11.h>

// Check for keyboard interrupt from python
inline void check_interrupts() {
  if (PyErr_CheckSignals() != 0) {
    throw pybind11::error_already_set();
  }
}

// Simple callback
template <typename real_t>
inline void update_progress(const uint64_t iter, const uint64_t maxIter,
                            const real_t eta, const real_t Eq,
                            const int write_per_it,
                            const unsigned long long int n_clashes) {
  fprintf(
      stderr,
      "%cOptimizing\t Progress: %.1lf%%, eta=%.4f, Eq=%.10f, clashes=%.1lf%%",
      13, (real_t)iter / maxIter * 100, eta, Eq,
      (real_t)n_clashes / (iter * write_per_it) * 100);
  fflush(stderr);
}

// Managed class
class ProgressMeter {
public:
  ProgressMeter(size_t total, bool percent = false)
      : total_(total), percent_(percent), count_(0) {
    tick(0);
  }

  void tick_count(size_t count) {
    count_ = count;
    if (percent_) {
      double progress = count_ / static_cast<double>(total_);
      progress = progress > 1 ? 1 : progress;
      fprintf(stderr, "%cProgress: %.1lf%%", 13, progress * 100);
    } else {
      size_t progress = count_ <= total_ ? count_ : total_;
      fprintf(stderr, "%cProgress: %lu / %lu", 13, progress, total_);
    }
  }

  void tick(size_t blocks) { tick_count(count_ + blocks); }

  void finalise() {
    if (percent_) {
      fprintf(stderr, "%cProgress: 100.0%%\n", 13);
    } else {
      fprintf(stderr, "%cProgress: %lu / %lu\n", 13, total_, total_);
    }
  }

private:
  size_t total_;
  bool percent_;
  volatile size_t count_;
};
