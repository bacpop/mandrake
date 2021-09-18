#pragma once

// C++ classes for CUDA graphs and streams

class cuda_graph {
public:
  cuda_graph() : graph_instance_(nullptr) {
    CUDA_CALL(cudaGraphCreate(&graph_, 0));
  }

  ~cuda_graph() {
    if (graph_ != nullptr) {
      CUDA_CALL_NOTHROW(cudaGraphDestroy(graph_));
    }
    if (graph_instance_ != nullptr) {
      CUDA_CALL_NOTHROW(cudaGraphExecDestroy(graph_instance_));
    }
  }

  cudaGraph_t &graph() { return graph_; }

  void launch(cudaStream_t stream) {
    if (graph_instance_ == nullptr) {
      CUDA_CALL(cudaGraphInstantiate(&graph_instance_, graph_, NULL, NULL, 0));
    }
    CUDA_CALL(cudaGraphLaunch(graph_instance_, stream));
  }

private:
  // Delete copy and move
  cuda_graph(const cuda_graph &) = delete;
  cuda_graph(cuda_graph &&) = delete;

  cudaGraph_t graph_;
  cudaGraphExec_t graph_instance_;
};

class cuda_stream {
public:
  cuda_stream() { CUDA_CALL(cudaStreamCreate(&stream_)); }

  ~cuda_stream() {
    if (stream_ != nullptr) {
      CUDA_CALL_NOTHROW(cudaStreamDestroy(stream_));
    }
  }

  cudaStream_t stream() { return stream_; }

  void capture_start() {
    CUDA_CALL(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
  }

  void capture_end(cudaGraph_t &graph) {
    cudaStreamEndCapture(stream_, &graph);
  }

  void add_host_fn(cudaHostFn_t fn, void *hostData) {
    CUDA_CALL(cudaLaunchHostFunc(stream_, fn, hostData));
  }

  void sync() { CUDA_CALL(cudaStreamSynchronize(stream_)); }

private:
  // Delete copy and move
  cuda_stream(const cuda_stream &) = delete;
  cuda_stream(cuda_stream &&) = delete;

  cudaStream_t stream_;
};
