cmake_minimum_required(VERSION 3.12)
project(SCE)
set(CMAKE_CXX_STANDARD 11)

# Variable definitions
set(TARGET_NAME SCE)
add_compile_definitions(PYTHON_EXT)

# Add -O0 to remove optimizations when using gcc
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -fopenmp -m64")
if(APPLE)
    set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin//clang")    
    set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin//clang")    
    set(CPP_FLAGS "${CPP_FLAGS} -I/usr/local/opt/llvm/include")
    set(LD_FLAGS "${LD_FLAGS} -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib")
endif()

# Add libraries
find_package(pybind11 REQUIRED)
find_package(GSL REQUIRED)
find_package(OpenMP REQUIRED)

# Check for CUDA
include(check_language)
check_language(CUDA)
if(CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "CUDA found, compiling CPU and GPU code")
    
    enable_language(CUDA)
    set_property(TARGET SCE 
                 PROPERTY CUDA_SEPARABLE_COMPILATION ON) 
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_AVAILABLE")
    set(CMAKE_CUDA_FLAGS "--cudart static -O3 --relocatable-device-code=false \
                          -gencode arch=compute_35,code=compute_35 -gencode \
                          arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50")
    
    add_compile_definitions(GPU_AVAILABLE) 
    set(TARGET_FILES src/wtsne_gpu.cu)
else()
    message(STATUS "CUDA not found, compiling CPU code only")
endif()


# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

set(TARGET_FILES "${TARGET_FILES}" src/wtsne_cpu.cpp src/python_bindings.cpp)

pybind11_add_module("${TARGET_NAME}" "${TARGET_FILES}")
target_link_libraries("${TARGET_NAME}" PRIVATE m GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX gfortran rt)
