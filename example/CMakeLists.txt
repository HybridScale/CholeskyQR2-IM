
#----- Cpu version -----#


#----- Cpu lookahead version -----#

#----- Cpu gram-schmidt with chol -----#
add_executable(gschol_cpu main.cpp)
target_link_libraries(gschol_cpu PRIVATE gscholcpu
                                ${Boost_LIBRARIES})
target_compile_definitions(gschol_cpu PRIVATE GSCHOL)

if(USE_GPU)
    #----- Gpu version -----#
    add_executable(cqr2_gpu main.cpp)
    target_link_libraries(cqr2_gpu PRIVATE cqr2bgs
                                    ${Boost_LIBRARIES})

    #----- Gpu lookahead version -----#
    add_executable(cqr2_gpu_lookahead main.cpp)
    target_link_libraries(cqr2_gpu_lookahead PRIVATE cqr2bgslookahead
                                                  ${Boost_LIBRARIES})
    target_compile_definitions(cqr2_gpu_lookahead PRIVATE LOOKAHEAD)

    #----- Shifted Cholesky version -----#
    add_executable(scqr3_gpu main.cpp)
    target_link_libraries(scqr3_gpu PRIVATE scqr3
                                                  ${Boost_LIBRARIES})
    target_compile_definitions(scqr3_gpu PRIVATE SHIFT)
    
    #----- Gpu version -----#
    add_executable(gschol_gpu main.cpp)
    target_link_libraries(gschol_gpu PRIVATE gschol
                                    ${Boost_LIBRARIES})
    target_compile_definitions(gschol_gpu PRIVATE GSCHOL)


        if(USE_NCCL)
            #----- Gpu version -----#
            add_executable(cqr2_gpu_nccl main.cpp)
            target_link_libraries(cqr2_gpu_nccl PRIVATE cqr2bgsNCCL
                                                        ${Boost_LIBRARIES})

            #----- GpuNCCL lookahead version -----#
            add_executable(cqr2_gpu_lookahead_nccl main.cpp)
            target_link_libraries(cqr2_gpu_lookahead_nccl PRIVATE cqr2bgslookaheadNCCL
                                                              ${Boost_LIBRARIES})
            target_compile_definitions(cqr2_gpu_lookahead_nccl PRIVATE LOOKAHEAD)

            #----- Shifter NCCL  version -----#
            add_executable(scqr3_gpu_nccl main.cpp)
            target_link_libraries(scqr3_gpu_nccl PRIVATE scqr3NCCL
                                                          ${Boost_LIBRARIES})
            target_compile_definitions(scqr3_gpu_nccl PRIVATE SHIFT)

            #----- Gpu version -----#
            add_executable(gschol_gpu_nccl main.cpp)
            target_link_libraries(gschol_gpu_nccl PRIVATE gscholNCCL
                                                          ${Boost_LIBRARIES})
            target_compile_definitions(gschol_gpu_nccl PRIVATE GSCHOL)

        endif()
endif()