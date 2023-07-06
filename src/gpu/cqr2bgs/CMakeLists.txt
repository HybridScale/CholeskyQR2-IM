
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

target_compile_definitions(cqr2bgs PRIVATE GPU)