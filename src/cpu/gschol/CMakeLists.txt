
add_library(gscholcpu SHARED gschol.cpp)
target_link_libraries(gscholcpu MPI::MPI_CXX 
                              BLAS::BLAS
                              validate
                              distributedmatrix)

target_include_directories(gscholcpu PUBLIC ${CMAKE_SOURCE_DIR}/include 
                                          ${CMAKE_SOURCE_DIR}/include/internals)