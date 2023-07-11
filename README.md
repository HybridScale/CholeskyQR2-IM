# Cholesky QR2 factorization
This is a repository that implements the [Cholesky QR2](https://ieeexplore.ieee.org/document/7016731) distributed algorithm, an alternative method to compute an orthogonal factor Q in the QR factorization of tall-and-skinny matrices.

## Getting Started


## Prerequisites

Prerequsites are listed here:

* CMake
* boost program options
* CUDA Toolkit
* BLAS
* MPI

### Other prerequisites

* NCCL (optional)


## How to compile
The compilation is done with `CMake`:

```bash
 cmake -B build && cmake --build build
```

On the Supek (Cray) system, the Cray compiler wrapper is needed instead of the direct compiler to link with the apropirate `CUDA AWARE MPICH` libraries. Therefore, pass this compiler wrapper to CMake with the `CMAKE_CXX_COMPILER` option. Additionally, provide the `BLA_VENDOR` option to instruct `CMAKE` to use `MKL` libraries instead of the `LibSci`, Cray Scientific Libraries package found by default.

```bash
source activate_supek_gnu.sh
cmake -B build-gnu -DBLA_VENDOR=Intel10_64ilp -DCMAKE_CXX_COMPILER=CC && cmake --build build-gnu
```

```bash
source activate_supek_nvhpc.sh
cmake -B build-nvhpc -DBLA_VENDOR=Intel10_64ilp -DCMAKE_CXX_COMPILER=CC && cmake --build build-nvhpc
```

The following table provides CMake build options:

| Description                | CMake Option | Supported Values | Default Value |
|----------------------------|--------------|------------------|---------------|
| build gpu versions         | USE_GPU      | True, False      | True          |
| use nccl for communication | USE_NCCL     | True, False      | True          |

Upon successuful completition the following binaries will be available from the build directory:

| Executable              |  Description                    |
|-------------------------|---------------------------------|
| cqr2_gpu                | gpu version (MPI blocking collective routines) |
| cqr2_gpu_nccl           | gpu version (NCCL  collective routines)        |
| cqr2_gpu_lookahead      | gpu version (MPI blocking collective routines) with lookahead method of updating rest of matrix |
| cqr2_gpu_lookahead_nccl | gpu version (NCCL collective routines) with lookahead method of updating rest of matrix |


## Usage
All builted executables have the same program options parameters, which can be displayed with the `-h` flag.

```bash
./executable -h
```

The next code block shows an example with simply explained options:

```bash
./executable --m <number of rows> --n <number of columns> --b <bgs panel size> --input <matrix>
```

## Additional Documentation and Acknowledgments

