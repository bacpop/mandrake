#pragma once

#include <cstddef> // size_t
#include <vector>

const int n_sample_frames = 200;

template <typename real_t> class sce_results {
public:
  sce_results(const bool make_animation, const uint64_t max_iter)
    : make_animation_(make_animation), sample_points_(n_sample_frames - 1), sample_it_(sample_points_.cbegin()), iter_series_(),
    eq_series_(), embedding_series_() {
      for (size_t i = 0; i < sample_points_.size(); ++i) {
        sample_points_[i] = std::round(max_iter * (1 - std::sqrt(1 - (double)(i + 1)/n_sample_frames)));
      }
    }

  void add_result(std::vector<real_t> &embedding) {
    embedding_series_.push_back(std::move(embedding));
  }

  void add_frame(const uint64_t iter, const real_t Eq, const std::vector<real_t> &embedding) {
    if (make_animation_ && iter >= *sample_it_) {
      iter_series_.push_back(iter);
      eq_series_.push_back(Eq);
      embedding_series_.push_back(embedding);
      if (++sample_it_ == sample_points_.cend()) {
        --sample_it_;
      }
    }
  }

  bool is_animated() const { return make_animation_; }
  size_t n_frames() const { return eq_series_.size(); }
  std::tuple<std::vector<uint64_t>, std::vector<real_t>> get_eq() const { return std::make_tuple(iter_series_, eq_series_); }

  // Get the result (last frame)
  std::vector<real_t> get_embedding() const {
    return embedding_series_.back();
  }

  // Get a specific frame (for animation)
  std::vector<real_t> get_embedding_frame(const size_t frame) const {
    if (frame < n_frames()) {
      return embedding_series_[frame];
    } else {
      throw std::runtime_error("Frame does not exist");
    }
  }

private:
  bool make_animation_;
  std::vector<uint64_t> sample_points_;
  std::vector<uint64_t>::const_iterator sample_it_;
  std::vector<uint64_t> iter_series_;
  std::vector<real_t> eq_series_;
  std::vector<std::vector<real_t>> embedding_series_;
};
