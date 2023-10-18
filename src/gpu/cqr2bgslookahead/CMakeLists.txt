
add_library(cqr2bgslookahead SHARED cqr2bgslookahead.cpp)
target_link_libraries(cqr2bgslookahead MPI::MPI_CXX 
                                       CUDA::cublas
                                       CUDA::cusolver
                                       CUDA::nvToolsExt
                                       cudamemory
                                       validate_GPU
                                       distributedmatrix)

target_include_directories(cqr2bgslookahead PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                                   ${CMAKE_SOURCE_DIR}/include/internals)
target_compile_definitions(cqr2bgslookahead PUBLIC GPU)

if(USE_NCCL)
    add_library(cqr2bgslookaheadNCCL SHARED cqr2bgslookahead.cpp)
    target_link_libraries(cqr2bgslookaheadNCCL MPI::MPI_CXX 
                                            CUDA::cublas
                                            CUDA::cusolver
                                            CUDA::nvToolsExt
                                            NCCL::NCCL
                                            cudamemory
                                            validate_GPU
                                            distributedmatrix)


    target_include_directories(cqr2bgslookaheadNCCL PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                                    ${CMAKE_SOURCE_DIR}/include/internals)
    target_compile_definitions(cqr2bgslookaheadNCCL PUBLIC GPU NCCL)

endif()

