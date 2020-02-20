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

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
endif()

# Check for CUDA and compiles GPU library
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, compiling both GPU and CPU code")
    
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "--cudart static --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50")
    
    add_compile_definitions(GPU_AVAILABLE)
    add_library("${TARGET_NAME}_CUDA" STATIC src/wtsne_gpu.cu)
    set_property(TARGET "${TARGET_NAME}_CUDA" 
                 PROPERTY POSITION_INDEPENDENT_CODE ON
                 #CUDA_SEPARABLE_COMPILATION ON
                 CUDA_VISIBILITY_PRESET "hidden")
else()
    message(STATUS "CUDA not found, compiling CPU code only")
endif()

# Compile CPU library
add_library("${TARGET_NAME}" MODULE src/wtsne_cpu.cpp src/python_bindings.cpp)
set_target_properties("${TARGET_NAME}" PROPERTIES 
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

#pybind11_add_module("${TARGET_NAME}" "${TARGET_FILES}")
if(CMAKE_CUDA_COMPILER)
    target_link_libraries("${TARGET_NAME}" PRIVATE "${TARGET_NAME}_CUDA")
endif()
target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module GSL::gsl GSL::gslcblas OpenMP::OpenMP_CXX gfortran m rt)

