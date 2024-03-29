cmake_minimum_required(VERSION 3.19)
project(SCE)
set(CMAKE_CXX_STANDARD 14)
cmake_policy(SET CMP0104 OLD) # Can't get CUDA_ARCHITECTURES to work with NEW

# Variable definitions
set(TARGET_NAME SCE)
add_compile_definitions(PYTHON_EXT)

IF(CMAKE_COMPILER_IS_GNUCC OR "$ENV{SCE_BUILD_ENV}" STREQUAL "conda_forge")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
# Set these to profile
# set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
# set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DUSE_CUDA_PROFILER")

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
    set(CMAKE_LD_FLAGS "${CMAKE_LDFLAGS} -Wl,--as-needed")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ffast-math -funroll-loops -m64")

# Set paths for non standard lib/ and include/ locations
if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
    link_directories($ENV{CONDA_PREFIX}/lib/intel64)
else()
  find_package(OpenMP)
endif()

# Add libraries
find_package(ZLIB)
find_package(pybind11 REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor)

# Check for CUDA and compiles GPU library
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA found, compiling both GPU and CPU code")

    enable_language(CUDA)
    set(CUDA_OPTS "-Wreorder -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -fopenmp -Xptxas -dlcm=ca --cudart static --relocatable-device-code=false")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        string(APPEND CUDA_OPTS " -G")
    else()
        string(APPEND CUDA_OPTS " --use_fast_math")
    endif()
    # Set this to profile
    # string(APPEND CUDA_OPTS " -pg --generate-line-info")
    set(CMAKE_CUDA_FLAGS "${CUDA_OPTS}")

    add_compile_definitions(GPU_AVAILABLE)
    add_library("${TARGET_NAME}_CUDA" OBJECT src/wtsne_gpu.cu)
    target_include_directories("${TARGET_NAME}_CUDA" PRIVATE "${pybind11_INCLUDE_DIRS}")
    set_property(TARGET "${TARGET_NAME}_CUDA"
                 PROPERTY POSITION_INDEPENDENT_CODE ON
                 CUDA_SEPARABLE_COMPILATION OFF
                 CUDA_RESOLVE_DEVICE_SYMBOLS ON   # try and ensure device link with nvcc
                 CUDA_VISIBILITY_PRESET "hidden")
else()
    message(STATUS "CUDA not found, compiling CPU code only")
endif()

# Compile CPU library
add_library("${TARGET_NAME}" MODULE src/wtsne_cpu.cpp src/python_bindings.cpp)
set_target_properties("${TARGET_NAME}" PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
if(UNIX AND (NOT APPLE OR NOT CMAKE_COMPILER_IS_GNUCC))
    set_target_properties("${TARGET_NAME}" PROPERTIES
                          INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries("${TARGET_NAME}" PRIVATE "${TARGET_NAME}_CUDA")
    set_property(TARGET "${TARGET_NAME}" PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

target_link_libraries("${TARGET_NAME}" PRIVATE pybind11::module)
if(DEFINED ENV{CONDA_PREFIX} AND (NOT APPLE OR CMAKE_COMPILER_IS_GNUCC OR ENV{SCE_BUILD_ENV} EQUAL "conda_forge"))
  target_link_libraries("${TARGET_NAME}" PRIVATE gomp z)
else()
  target_link_libraries("${TARGET_NAME}" PRIVATE ZLIB::ZLIB)
  if(OpenMP_CXX_FOUND)
      target_link_libraries("${TARGET_NAME}" PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
