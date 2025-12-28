cmake_minimum_required(VERSION 3.18)
project(softmax_cpu_Kozhevatov)

enable_language(CUDA)

set(target_suffix Kozhevatov)
set(target_name "softmax_cpu_${target_suffix}")

add_executable(${target_name} main.cu)

target_compile_features(${target_name} PRIVATE cxx_std_17)

set_target_properties(${target_name}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "75"
)

# Добавляем флаги CUDA
target_compile_options(${target_name} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${target_name} PRIVATE CUDA::cudart)

# CUTLASS
target_include_directories(${target_name} PRIVATE "/opt/cutlass/include")