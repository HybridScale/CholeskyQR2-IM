
add_library(scqr3 SHARED scqr3.cpp)
target_link_libraries(scqr3 MPI::MPI_CXX 
                              CUDA::cublas
                              CUDA::cusolver
                              CUDA::nvToolsExt
                              cudamemory
                              validate
                              distributedmatrix)

target_include_directories(scqr3 PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                          ${CMAKE_SOURCE_DIR}/include/internals)

target_compile_definitions(scqr3 PUBLIC GPU)


if(USE_NCCL)
    add_library(scqr3NCCL SHARED scqr3.cpp)
    target_link_libraries(scqr3NCCL MPI::MPI_CXX 
                                      CUDA::cublas
                                      CUDA::cusolver
                                      CUDA::nvToolsExt
                                      NCCL::NCCL
                                      cudamemory
                                      validate
                                      distributedmatrix)


    target_include_directories(scqr3NCCL PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                                  ${CMAKE_SOURCE_DIR}/include/internals)
    target_compile_definitions(scqr3NCCL PUBLIC GPU NCCL)

endif()