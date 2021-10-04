#pragma once

#include <cstddef> // size_t
#include <cstdint>
#include <cstdio>

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
