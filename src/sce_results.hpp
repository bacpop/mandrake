#pragma once

#include <cstddef> // size_t
#include <vector>

const int n_sample_frames = 400;

template <typename real_t> class sce_results {
public:
  sce_results(const bool make_animation, const size_t n_workers,
              const uint64_t max_iter)
      : make_animation_(make_animation), n_workers_(n_workers),
        sample_points_(n_sample_frames - 1),
        sample_it_(sample_points_.cbegin()), iter_series_(), eq_series_(),
        embedding_series_() {
    sample_points_[0] = 0;
    for (size_t i = 1; i < sample_points_.size(); ++i) {
      sample_points_[i] = std::round(
          max_iter * (1 - std::sqrt(1 - (double)i / n_sample_frames)));
    }
  }

  void add_result(const uint64_t iter, const real_t Eq,
                  std::vector<real_t> &embedding) {
    embedding_series_.push_back(std::move(embedding));
    if (make_animation_) {
      iter_series_.push_back(iter * n_workers_);
      eq_series_.push_back(Eq);
    }
  }

  bool is_sample_frame(const uint64_t iter) const {
    return make_animation_ && sample_it_ != sample_points_.cend() &&
           iter >= *sample_it_;
  }

  void add_frame(const uint64_t iter, const real_t Eq,
                 const std::vector<real_t> &embedding) {
    if (is_sample_frame(iter)) {
      iter_series_.push_back(iter * n_workers_);
      eq_series_.push_back(Eq);
      embedding_series_.push_back(embedding);
      sample_it_++;
    }
  }

  bool is_animated() const { return make_animation_; }
  size_t n_frames() const { return eq_series_.size(); }
  std::tuple<std::vector<uint64_t>, std::vector<real_t>> get_eq() const {
    return std::make_tuple(iter_series_, eq_series_);
  }

  // Get the result (last frame)
  std::vector<real_t> get_embedding() const { return embedding_series_.back(); }

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
  size_t n_workers_;
  std::vector<uint64_t> sample_points_;
  std::vector<uint64_t>::const_iterator sample_it_;
  std::vector<uint64_t> iter_series_;
  std::vector<real_t> eq_series_;
  std::vector<std::vector<real_t>> embedding_series_;
};
