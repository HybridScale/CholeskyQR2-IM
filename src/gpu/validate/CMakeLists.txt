add_library(validate_GPU SHARED validate_gpu.cpp)
target_link_libraries(validate_GPU CUDA::cublas 
                                   MPI::MPI_CXX
                                   NCCL::NCCL)
target_include_directories(validate_GPU PUBLIC ${CMAKE_SOURCE_DIR}/include/internals)
target_compile_definitions(validate_GPU PRIVATE NCCL)