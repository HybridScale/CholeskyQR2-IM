ml utils/intel-oneapi-compilers/2023.1.0

ml PrgEnv-nvhpc/8.3.3
ml cray-pals
ml utils/intel-oneapi-mkl/2023.1.0
ml libs/cuda/11.6

spack load boost

export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/comm_libs/nccl:$CMAKE_PREFIX_PATH

#instead of manually setting env variable CRAY_ACCLE_TARGET lloading craype-accel-nvidia80
#export CRAY_ACCEL_TARGET=nvidia80
ml craype-accel-nvidia80

export MPICH_GPU_SUPPORT_ENABLED=1