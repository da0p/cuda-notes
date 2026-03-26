cmake_minimum_required(VERSION 3.18)
project(cuda_exp CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Optional: set default arch, or use native to auto-detect
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(add_cuda
  add_cuda.cu
)

target_link_libraries(add_cuda
  CUDA::cudart
)

add_executable(basic_stream
  basic_stream.cu
)

target_link_libraries(basic_stream
  CUDA::cudart
)

add_executable(advanced_stream
  advanced_stream.cu
)

target_link_libraries(advanced_stream
  CUDA::cudart
)

add_executable(cu_std_atomic 
  cu_std_atomic.cu
)

target_link_libraries(cu_std_atomic
  CUDA::cudart
)

target_include_directories(cu_std_atomic
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}/../cccl
)

add_executable(gpu_transform
  gpu_transform.cu
)

target_link_libraries(gpu_transform
  CUDA::cudart
)

target_include_directories(gpu_transform
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}/../cccl
)

target_compile_options(gpu_transform
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(zip_iterator
  zip_iterator.cu
)

target_link_libraries(zip_iterator
  CUDA::cudart
)

target_include_directories(zip_iterator
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}/../cccl
)

target_compile_options(zip_iterator
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

add_executable(transform_iterator
  transform_iterator.cu
)

target_link_libraries(transform_iterator
  CUDA::cudart
)

target_include_directories(transform_iterator
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}/../cccl
)

target_compile_options(transform_iterator
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
