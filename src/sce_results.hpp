#pragma once

#include <cstddef> // size_t
#include <vector>

// TODO: the idea here is that the python bindings will wrap this class,
// and these methods can be called directly from python to make the plots there
// Use a smart pointer to create the object from python, then pass it to the
// wtsne function

template <typename real_t> class sce_results {
public:
  sce_results(const bool make_animation)
      : make_animation_(make_animation), iter_series_(), eq_series_(), embedding_series_() {}

  void add_result(std::vector<real_t> &embedding) {
    embedding_series_.push_back(std::move(embedding));
  }

  void add_frame(const uint64_t iter, const real_t Eq, const std::vector<real_t> &embedding) {
    if (make_animation_) {
      iter_series_.push_back(iter);
      eq_series_.push_back(Eq);
      embedding_series_.push_back(embedding);
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
  std::vector<uint64_t> iter_series_;
  std::vector<real_t> eq_series_;
  std::vector<std::vector<real_t>> embedding_series_;
};
