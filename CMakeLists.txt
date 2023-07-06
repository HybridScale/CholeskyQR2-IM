cmake_minimum_required(VERSION 3.17)

project(cholesky_qr2 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(USE_GPU "Use NVIDIA collective communication library" TRUE)
if (USE_GPU)
    option(USE_NCCL "Use NVIDIA collective communication library" TRUE)
endif()

find_package(BLAS REQUIRED )
find_package(LAPACK REQUIRED )
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(MPI REQUIRED)

if(USE_GPU)
    find_package(CUDAToolkit REQUIRED )
endif()

if(USE_NCCL)
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    find_package(NCCL REQUIRED )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_subdirectory("src/")
add_subdirectory("example")