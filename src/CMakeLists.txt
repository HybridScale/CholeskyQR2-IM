
add_subdirectory("validate")
add_subdirectory("distributedmatrix")

#add_subdirectory("cpu/cqr2bgs")
#add_subdirectory("cpu/cqr2bgslookahead")

if(USE_GPU)
    add_subdirectory("gpu/cudamemory")
    add_subdirectory("gpu/cqr2bgs")
    add_subdirectory("gpu/cqr2bgslookahead")
endif()