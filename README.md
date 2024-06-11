![GitHub License](https://img.shields.io/github/license/HybridScale/CholeskyQR2-IM) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10888693.svg)](https://doi.org/10.5281/zenodo.10888693)

# CholeskyQR2 for ill-conditioned matrices

The **CholeskyQR2** for **I**ll-conditioned **M**atrices (**CholeskyQR2-IM**) library is a modern C++ implementation designed to compute the QR factorization of **extremely ill-conditioned tall-and-skinny** matrices. It is based on the [Cholesky QR2](https://ieeexplore.ieee.org/document/7016731) algorithm, which has been widely recognized for its performance and easy parallelization on the distributed memory systems.

The **CholeskyQR2-IM** introduces an alternative approach that combines traditional CholeskyQR2 with the **shifting** technique and the **modified Gram-Schmidt** process. The innovative approach enhances both the **numerical stability** of the algorithm and the **accuracy** of the computed factor Q.

The library is specifically designed for QR factorisation of large tall-and-skinny matrices on distributed memory systems, with full support for both GPUs and modern CPUs. Comprehensive insights into the novel QR algorithm as well as detailed analyses of its stability and performance can be found [here](https://arxiv.org/abs/2405.04237).

## Algorithmic variants

1. **cqr2**: This version integrates the CholeskyQR algorithm with a modified Gram-Schmidt (MGS) process, which orthogonalizes the columns to the right of the current panel. The CholeskyQR with MGS is performed twice (CholeskyQR2) to enhance orthogonality. In cases of extremely ill-conditioned matrices (*cond(A)>10^8*), increasing the number of panels is necessary (refer to the [paper]()).

2. **scqr3**: This version adopts the shifting technique instead of MGS to enhance numerical stability. The code is based on the work of [Fukaya et al.](https://epubs.siam.org/doi/abs/10.1137/18M1218212).The shifted CholeskyQR initially serves as a preconditioner for CholeskyQR2, necessitating three repetitions of CholeskyQR. Due to the increased number of floating-point operations (flops), this version proves to be the slowest approach in our benchmarks compared to the other two variants.

3. **gschol**: This version is the refinement of the **cqr2** variant. Instead of applying CholeskyQR twice, where the first factor Q may not be fully orthogonal, each panel in this version undergoes complete orthogonalization before applying MGS to subsequent panels to the right. In our benchmarks, this variant demostrated to be the fastest and numerically stable for matrices with condition number up to 10^15.

## Quick start

### Prerequisites

Dependencies:

* C++ compiler
* CMake
* Boost (program options)
* CUDA Toolkit (CuBLAS, CuSolver)
* BLAS library
* MPI (CUDA-aware) 
* NCCL (optional)

On the supercomputer Supek at the University Computing Centre, University of Zagreb, the CholeskyQR2-IM library was compiled with the following dependencies:

- gcc@11.2.0
- CMake@3.22.2
- Boost@1.78.0
- CUDA@11.4.2
- IntelMKL@2020.4.304
- OpenMPI@4.1.2
- nccl@2.11.4

### How to compile
The compilation is done with `CMake`:

```bash
 cmake -B build && cmake --build build
``` 

The following table provides additional CMake build options:

| Description                | CMake Option | Supported Values | Default Value |
|----------------------------|--------------|------------------|---------------|
| build with support for GPU        | USE_GPU      | True, False      | True          |
| use nccl for communication | USE_NCCL     | True, False      | True          |

By default the library will be compiled with GPU support and using NCCL for collective communication. 


Below are the examples on how to compile the library with specific supports:

- **GPU with CUDA-aware MPI**
```bash
 cmake -B build -DUSE_GPU=1 -DUSE_NCCL=0 && cmake --build build
``` 

- **CPU-only with MPI**
```bash
 cmake -B build -DUSE_GPU=0 && cmake --build build
``` 

*NOTE: Currently, only the **gschol** algorithmic variant is available in the CPU-only mode*

Upon successful completion (using the default compilation options), the following tester executables will be available in the build directory:

| Executable              |  Architecture | Communicator |  Lookahead |
|-------------------------|---------------|--------------|------------|
| cqr2_gpu                | gpu           | MPI          | no         |
| cqr2_gpu_lookahead      | gpu           | MPI          | yes        |
| scqr3_gpu               | gpu           | MPI          | no         |
| gschol_gpu              | gpu           | MPI          | no         |
| gschol_cpu              | cpu           | MPI          | no         |
| cqr2_gpu_nccl           | gpu           | NCCL         | no         |
| cqr2_gpu_lookahead_nccl | gpu           | NCCL         | yes        |
| scqr3_gpu_nccl          | gpu           | NCCL         | no         |
| gschol_gpu_nccl         | gpu           | NCCL         | no         |

### Usage

All built executables share the same program options, which can be displayed using the `-h` flag.

```bash
./<executable> -h
```

The following code block illustrates an example with easily explained options:

```bash
./<executable> --m <number of rows> --n <number of columns> --b <panel size> --input <path-to-matrix>
```

## Acknowledgments

The code was developed at the [Ruđer Bošković Institute (RBI)](https://www.irb.hr/) in the [Centre for Informatics and Computing](https://www.irb.hr/eng/Scientific-Support-Centres/Centre-for-Informatics-and-Computing). The research and development of the library was supported by the Croatia Science Foundation through the project UIP-2020-02-4559 "Scalable High-Performance Algorithms for Future Heterogeneous Distributed Computing Systems ([HybridScale](https://www.croris.hr/projekti/projekt/6243?lang=en))".

## Developers

[Nenad Mijić](https://github.com/Nenad03), RBI, Croatia

[Abhiram Kaushik](https://github.com/abhiramkb), RBI, Croatia

[Davor Davidović](https://github.com/ddavidovic), RBI, Croatia

## How to cite

To cite the CholeskyQR2-IM please use the following paper:

- N. Mijić, A. Kaushik Badrinarayanan, D. Davidović. QR factorization of ill-conditioned tall-and-skinny matrices on distributed-memory systems, Arxiv, 2024, 2405.04237 ([here](https://arxiv.org/abs/2405.04237))

## Copyright and License

This code is published under 3-Clause BSD License ([BSD License 2.0](./LICENSE))