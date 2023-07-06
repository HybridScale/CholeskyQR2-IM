
add_library(distributedmatrix SHARED distributedmatrix.cpp)
target_link_libraries(distributedmatrix MPI::MPI_CXX)
target_include_directories(distributedmatrix PUBLIC ${CMAKE_SOURCE_DIR}/include/internals)