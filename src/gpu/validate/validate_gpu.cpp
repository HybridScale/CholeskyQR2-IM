/*
 * File:   validate_gpu.cpp
 * Date:   July 7, 2023
 * Brief:  Implementation of validation functions for GPU. Computes the residual and orthogonality of the obtained factors Q and R.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */

#include <cmath>
#include <iostream>
#include <fstream>

#include <string>
#include "validate_gpu.hpp"


double Validate::orthogonality()
{
    double norm = -14.0;
    double alpha = 1.0, beta = 0.0;

    cudamemory<double> C(n_ * n_);

    cublasDgemm(cublashandle_,
                CUBLAS_OP_N, CUBLAS_OP_T,
                n_, n_, m_,
                &alpha,
                Q_, n_,
                Q_, n_,
                &beta,
                C.data(), n_);

    ncclAllReduce(C.data(), C.data(), n_*n_, ncclDouble, ncclSum, nccl_comm_, 0);

    cudamemory<double> I(n_);
    I.memset(1.0);

    alpha = -1.0;
    std::int64_t incx = 1, incy = n_+1;

    cublasDaxpy(cublashandle_, n_,
                &alpha,
                I.data(), incx,
                C.data(), incy);

    std::int64_t n2 = n_ * n_;
    cublasDnrm2(cublashandle_, n2,
                C.data(), incx, &norm);
    return (norm / std::sqrt(n_));
}


double Validate::residuals(cudamemory<double> &A)
{
    double norm = -14.0;
    double normA = -14.0;

    double alpha = 1.0;
    double beta = 0.0;

    std::int64_t incx = 1; 

    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                             n_, m_,
                             &alpha,
                             R_, n_,
                             Q_, n_,
                             Q_, n_));

 
    cublasDnrm2(cublashandle_, size_,
                A.data(), incx, &normA);

    normA *= normA;
    MPI_Allreduce(MPI_IN_PLACE, &normA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    normA = std::sqrt(normA);
        
    alpha = -1.0;
    cublasDaxpy(cublashandle_, size_,
                &alpha,
                A.data(), incx,
                Q_, incx);
    
    cublasDnrm2(cublashandle_, size_,
                Q_, incx, &norm);

    norm *= norm;
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    norm = std::sqrt(norm);

    return (norm/normA);            

}