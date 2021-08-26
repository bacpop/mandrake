#pragma once

#include <vector>

#include "cuda_call.cuh"

template <typename T>
class device_value {
public:
  // Default constructor
  device_value() {
    CUDA_CALL(cudaMalloc((void**)&data_, sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, sizeof(T)));
  }

  // Constructor from value
  device_value(const T value) {
    CUDA_CALL(cudaMalloc((void**)&data_, sizeof(T)));
    CUDA_CALL(cudaMemset(data_, value, sizeof(T)));
  }

  // Copy
  device_value(const device_value& other) {
    CUDA_CALL(cudaMalloc((void**)&data_, sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, other.data_, sizeof(T),
                         cudaMemcpyDefault));
  }

  // Copy assign
  device_value& operator=(const device_value& other) {
    if (this != &other) {
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMalloc((void**)&data_, sizeof(T)));
      CUDA_CALL(cudaMemcpy(data_, other.data_, sizeof(T),
                           cudaMemcpyDefault));
    }
    return *this;
  }

  // Move
  device_value(device_value&& other) : data_(nullptr) {
    data_ = other.data_;
    other.data_ = nullptr;
  }

  // Move assign
  device_value& operator=(device_value&& other) {
    if (this != &other) {
      CUDA_CALL(cudaFree(data_));
      data_ = other.data_;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~device_value() {
    CUDA_CALL_NOTHROW(cudaFree(data_));
  }

  T get_value() const {
    T* host_value;
    CUDA_CALL(cudaMemcpy(T, data_, sizeof(T),
                        cudaMemcpyDefault));
    return *T;
  }

  void set_array(const T value) {
    CUDA_CALL(cudaMemcpy(data_, &value, sizeof(T),
                        cudaMemcpyDefault));
  }

  T* data() {
    return data_;
  }

private:
  T* data_;
};

template <typename T>
class device_array {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
  }

  // Copy
  device_array(const device_array& other) : size_(other.size_) {
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                         cudaMemcpyDefault));
  }

  // Copy assign
  device_array& operator=(const device_array& other) {
    if (this != &other) {
      size_ = other.size_;
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                           cudaMemcpyDefault));
    }
    return *this;
  }

  // Move
  device_array(device_array&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  // Move assign
  device_array& operator=(device_array&& other) {
    if (this != &other) {
      CUDA_CALL(cudaFree(data_));
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
    CUDA_CALL_NOTHROW(cudaFree(data_));
  }

  void get_array(std::vector<T>& dst) const {
    CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
                        cudaMemcpyDefault));
  }

  void set_array(const std::vector<T>& src) {
    size_ = src.size();
    CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                        cudaMemcpyDefault));
  }

  void set_array(const T* src) {
    CUDA_CALL(cudaMemcpy(data_, src, size_ * sizeof(T),
                        cudaMemcpyDefault));
  }

  T* data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

private:
  T* data_;
  size_t size_;
};

// Specialisation of the above for void* memory needed by some cub functions
// Construct once and use set_size() to modify
// Still using malloc/free instead of new and delete, as void type problematic
template <>
class device_array<void> {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {}
  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    if (size_ > 0) {
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
    }
  }

  ~device_array() {
    CUDA_CALL_NOTHROW(cudaFree(data_));
  }

  void set_size(size_t size) {
    size_ = size;
    CUDA_CALL(cudaFree(data_));
    if (size_ > 0) {
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
    } else {
      data_ = nullptr;
    }
  }

  void* data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

private:
  device_array ( const device_array<void> & ) = delete;
  device_array ( device_array<void> && ) = delete;

  void* data_;
  size_t size_;
};
