
add_library(cqr2bgs SHARED cqr2bgs.cpp)
target_link_libraries(cqr2bgs MPI::MPI_CXX 
                              CUDA::cublas
                              CUDA::cusolver
                              CUDA::nvToolsExt
                              cudamemory
                              validate
                              distributedmatrix)

target_include_directories(cqr2bgs PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                          ${CMAKE_SOURCE_DIR}/include/internals)

target_compile_definitions(cqr2bgs PUBLIC GPU)


if(USE_NCCL)
    add_library(cqr2bgsNCCL SHARED cqr2bgs.cpp)
    target_link_libraries(cqr2bgsNCCL MPI::MPI_CXX 
                                      CUDA::cublas
                                      CUDA::cusolver
                                      CUDA::nvToolsExt
                                      NCCL::NCCL
                                      cudamemory
                                      validate
                                      distributedmatrix)


    target_include_directories(cqr2bgsNCCL PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                                  ${CMAKE_SOURCE_DIR}/include/internals)
    target_compile_definitions(cqr2bgsNCCL PUBLIC GPU NCCL)

endif()