cmake_minimum_required(VERSION 3.24)

project(stim_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(stim_test stim_test.cu)
add_executable(l4_test l4_test.cu)
add_executable(v1_test adex_v1.cu)

foreach(_t stim_test l4_test v1_test)
    set_target_properties(${_t} PROPERTIES
        CUDA_ARCHITECTURES "86;89"
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(${_t} PRIVATE CUDA::cudart CUDA::curand)
    target_compile_options(${_t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endforeach()
