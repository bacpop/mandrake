#pragma once

static void HandleCUDAError(const char *file, int line,
                            cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
  cudaDeviceSynchronize();
#endif

if (status != cudaSuccess || (status = cudaGetLastError()) != cudaSuccess) {
  if (status == cudaErrorUnknown) {
    printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
  }
    printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line,
    cudaGetErrorString(status));
    throw std::runtime_error("CUDA error");
  }
}

#define CUDA_CALL(err) (HandleCUDAError(__FILE__, __LINE__, err))
#define CUDA_CALL_NOTHROW( err ) (err)
