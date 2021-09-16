#pragma once

#include <cstddef> // size_t
#include <vector>

template <typename real_t> class animate {
public:
  animate(const bool make_animation)
      : make_animation_(make_animation), eq_series_(0), embedding_series_(0) {}

  void add_frame(const real_t Eq, const std::vector<real_t> &embedding) {
    if (make_animation_) {
      eq_series_.push_back(Eq);
      embedding_series_.push_back(embedding);
    }
  }

  size_t n_frames() const { return eq_series_.size(); }
  std::vector<real_t> get_all_eq() const { return eq_series_; }
  std::vector<std::vector<real_t>> get_all_embedding() const {
    return embedding_series_;
  }

  real_t get_eq(const size_t frame) const {
    if (frame < n_frames()) {
      return eq_series_[frame];
    } else {
      throw std::runtime_error("Frame does not exist");
    }
  }

  std::vector<real_t> get_embedding(const size_t frame) const {
    if (frame < n_frames()) {
      return embedding_series_[frame];
    } else {
      throw std::runtime_error("Frame does not exist");
    }
  }

private:
  bool make_animation_;
  std::vector<real_t> eq_series_;
  std::vector<std::vector<real_t>> embedding_series_;
};
