
add_library(validate SHARED validate.cpp)
target_link_libraries(validate BLAS::BLAS)
target_include_directories(validate PUBLIC ${CMAKE_SOURCE_DIR}/include/internals)