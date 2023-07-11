ml PrgEnv-gnu/8.3.3
ml cray-pals
ml libs/boost/1.81.0-gnu
ml utils/intel-oneapi-mkl/2023.1.0
ml libs/cuda/11.6

spack load intel-oneapi-compilers@2023.0.0

export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/comm_libs/nccl:$CMAKE_PREFIX_PATH
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1