
add_library(cudamemory SHARED cudamemory.cpp)
target_link_libraries(cudamemory PUBLIC CUDA::cudart)
target_include_directories(cudamemory PUBLIC ${CMAKE_SOURCE_DIR}/include/internals)