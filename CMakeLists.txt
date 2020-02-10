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
    set(LD_FLAGS "${LD_FLAGS} -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib")
endif()

# Add libraries
find_package(pybind11 REQUIRED)

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

pybind11_add_module("${TARGET_NAME}" src/wtsne_cpu.cpp)
target_link_libraries("${TARGET_NAME}" PRIVATE m gsl gslcblas gomp)