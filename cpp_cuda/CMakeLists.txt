cmake_minimum_required(VERSION 3.24)

project(expectation_snn_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(expectation_snn_cuda_core STATIC
  src/richter_eval.cu
)

target_include_directories(expectation_snn_cuda_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(expectation_snn_cuda_core PUBLIC cxx_std_17)
target_compile_options(expectation_snn_cuda_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(expectation_snn_cuda_core PUBLIC CUDA::cudart)
set_target_properties(expectation_snn_cuda_core PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(expectation_snn_native
  src/native_cli.cpp
)

target_link_libraries(expectation_snn_native PRIVATE
  expectation_snn_cuda_core
)

target_compile_features(expectation_snn_native PRIVATE cxx_std_17)
set_target_properties(expectation_snn_native PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE PYBIND11_CMAKEDIR_STATUS
  OUTPUT_VARIABLE PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT PYBIND11_CMAKEDIR_STATUS EQUAL 0)
  message(FATAL_ERROR "pybind11 is required to build the native CUDA module")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native_cuda NO_EXTRAS
  src/bindings.cpp
)

target_link_libraries(_native_cuda PRIVATE
  expectation_snn_cuda_core
)

target_compile_features(_native_cuda PRIVATE cxx_std_17)
set_target_properties(_native_cuda PROPERTIES
  CUDA_ARCHITECTURES 89
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
