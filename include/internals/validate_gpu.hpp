/*
 * File:   validate_gpu.hpp
 * Date:   July 7, 2023
 * Brief:  Definition of validation functions for GPU. Computes the residual and orthogonality of the obtained factors Q and R.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nccl.h>

#include "mpi.h"

#include "cudamemory.hpp"
#include "utils.hpp"


class Validate
{
    private:
        std::int64_t m_, n_, size_;
        double* Q_;
        double* R_;
        const char *filename_;
        int world_rank_, world_size_;
        
        cublasHandle_t cublashandle_;

        ncclComm_t nccl_comm_;
        ncclUniqueId NCCLid_;

    public:
        Validate(std::int64_t m, std::int64_t n, double *Q, double *R, const char *file, cublasHandle_t cublashandle):
         m_(m), n_(n), Q_(Q), R_(R), filename_(file), cublashandle_(cublashandle)
         {
            size_ = m_ * n_;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

            if (world_rank_ == 0) 
                ncclGetUniqueId(&NCCLid_);

            MPI_Bcast(&NCCLid_, sizeof(NCCLid_), MPI_BYTE, 0, MPI_COMM_WORLD);
            ncclCommInitRank(&nccl_comm_, world_size_, NCCLid_, world_rank_);
         };

        double orthogonality();
        double residuals(cudamemory<double> &A);

};