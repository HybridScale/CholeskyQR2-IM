
add_library(gschol SHARED gschol.cpp)
target_link_libraries(gschol MPI::MPI_CXX 
                              CUDA::cublas
                              CUDA::cusolver
                              CUDA::nvToolsExt
                              cudamemory
                              validate_GPU
                              distributedmatrix)

target_include_directories(gschol PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                          ${CMAKE_SOURCE_DIR}/include/internals)

target_compile_definitions(gschol PUBLIC GPU)


if(USE_NCCL)
    add_library(gscholNCCL SHARED gschol.cpp)
    target_link_libraries(gscholNCCL MPI::MPI_CXX 
                                      CUDA::cublas
                                      CUDA::cusolver
                                      CUDA::nvToolsExt
                                      NCCL::NCCL
                                      cudamemory
                                      validate_GPU
                                      distributedmatrix)


    target_include_directories(gscholNCCL PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                                  ${CMAKE_SOURCE_DIR}/include/internals)
    target_compile_definitions(gscholNCCL PUBLIC GPU NCCL)

endif()