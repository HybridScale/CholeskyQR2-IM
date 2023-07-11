# Cholesky QR2 factorization
This is a repository that implements the [Cholesky QR2](https://ieeexplore.ieee.org/document/7016731) distributed algorithm, an alternative method to compute an orthogonal factor Q in the QR factorization of tall-and-skinny matrices.

## Getting Started


## Prerequisites

Prerequsites are listed here:

* cmake
* boost program options
* cuda
* BLAS
* MPI
* NCCL (optional)

### Other prerequisites


## How to compile
```bash
 cmake -B build && cmake --build build
```

On Supek system
```bash
cmake -B build -DBLA_VENDOR=Intel10_64ilp -DCMAKE_CXX_COMPILER=CC
```
The following table provides CMake build options:

| Description                | CMake Option | Supported Values | Default Value |
|----------------------------|--------------|------------------|---------------|
| build gpu versions         | USE_GPU      | True, False      | True          |
| use nccl for communication | USE_NCCL     | True, False      | True          |

Upon successuful completition the following binaries will be available from the build directory:

* cqr2_gpu_lookahead -> gpu version (MPI blocking collective routines) with lookahead method of updating rest of matrix
* cqr2_gpu_lookahead_nccl -> gpu version (NCCL collective routines) with lookahead method of updating rest of matrix


## Usage
```bash
./executable --m <number of rows> --n <number of columns> --input <matrix>
```

### Branches

## Additional Documentation and Acknowledgments

