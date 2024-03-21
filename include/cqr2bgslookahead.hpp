/*
 * File:   cqr2bgslookahead.hpp
 * Date:   July 7, 2023
 * Brief:  Definition of the class for CholeskyQR2 with modified block Gram-Schmidt reorthogonalization algorithm,
 *         including lookahead optimization.
 * 
 * This file is part of the CholeskyQR2++ library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <iostream>
#include <vector>
#include <string>
#include <memory>

#ifdef GPU
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    #include <cusolverDn.h>
#endif

#include "internals/distributedmatrix.hpp"
#include "internals/timing.hpp"

#ifdef GPU
    #include "internals/cudamemory.hpp"
    #include "internals/utils.hpp"
    #include "internals/validate_gpu.hpp"
#else
    #include "internals/validate.hpp"
#endif

#pragma once

namespace cqr
{

    class qr2bgsloohahead{
    public:
        qr2bgsloohahead(std::int64_t m, std::int64_t n, std::int64_t panel_size);
        qr2bgsloohahead(std::int64_t m, std::int64_t n, std::size_t panel_num);
        ~qr2bgsloohahead();

        void InputMatrix(cudamemory<double> &A);
        void InputMatrix(double *A);
        void InputMatrix(const char* filename);

        void Start();
        void Validate_output();

     private:
#ifdef GPU
        void cqr2bgs(cudamemory<double> &A, cudamemory<double> &R);
        void cqrbgs(cudamemory<double> &A, cudamemory<double> &R);
#else
        void cqr2bgs(std::vector<double> &A, std::vector<double> &R);
        void cqrbgs(std::vector<double> &A, std::vector<double> &R);
#endif
        void MPI_Warmup();
        void gramMatrix(double *A, double *R, double *tmp);

        void gramMatrixGemm(double *A, double *R, double *tmp, cudaStream_t stream);
        void gramMatrixCommunication(double *tmp, cudaStream_t stream);
        void gramMatrixRest(double *A, double *R, double *tmp, cudaStream_t stream);
        
        void cholesky(double *B, cudaStream_t stream);
        void calculateQ(double *A, double *R, cudaStream_t stream);
        void updateMatrix(std::int64_t m, std::int64_t n, std::int64_t k, 
                          double *A, std::int64_t lda,
                          double *B, std::int64_t ldb,
                          double *C, std::int64_t ldc,
                          double *Tmp, std::int64_t ldw,
                          cudaStream_t stream);
        
        void updateMatrixGemm(std::int64_t m, std::int64_t n, std::int64_t k, 
                              double *A, std::int64_t lda,
                              double *B, std::int64_t ldb,
                              double *Tmp, std::int64_t ldw,
                              cudaStream_t stream);
        void updateMatrixCommunication(double *Tmp, std::int64_t ldw, 
                                       cudaStream_t stream);
        void updateMatrixRest(std::int64_t m, std::int64_t n, std::int64_t k, 
                              double *A, std::int64_t lda,
                              double *B, std::int64_t ldb,
                              double *C, std::int64_t ldc,
                              double *Tmp, std::int64_t ldw,
                              cudaStream_t stream);
        
        float get_time();

        std::int64_t n_;                 // number of columns of the input matrix
        std::int64_t m_;                 // (global) number of rows of the input matrix
        std::int64_t localm_;            // local number of rows per MPI rank
        std::int64_t input_panel_size_;  // panel width (number of columns in the panel)
        std::int64_t panel_size_;        // local variable for keeping current panel size
        std::int64_t size = 1;           // local variable, total number of elements of the input matrix (m_ * n_)
        std::string filename_;           // name of the input file

        std::vector<double> A_;          // Global array for storing input matrix
        std::vector<double> Alocal_;     // Local array for storing a block of A (per MPI rank)
        std::vector<double> R_;          // Local array for string R factor

#ifdef GPU
        cudamemory<double> cudaAlocal_;
        cudamemory<double> cudaR_, cudaR1_, cudaR2_;
        cudamemory<double> cudaI_;
        cudamemory<double> cudatmp_, cudatmpqr2_;
        cudamemory<double> cudaWtmp1_, cudaWtmp2_;

        cudamemory<double> d_work;
        cudamemory<int> d_info;
#endif

        std::unique_ptr<DistributedMatrix> distmatrix;
        std::unique_ptr<Validate> validate;
#ifdef GPU
        std::unique_ptr<TimingGpu> timing;
#endif

        double orthogonality_, residuals_;

        float time_of_execution_ = 0;

        MPI_Comm mpi_comm_;
        MPI_Comm shmcomm_;
        int world_size_;
        int world_rank_;
        int shmsize_;
        int shmrank_;

#ifdef GPU
        cublasHandle_t cublashandle_;
        cusolverDnHandle_t cusolverhandle_;

        cudaStream_t stream_panelqr1, stream_panelqr2;
        cudaStream_t stream_update1, stream_update2;

    #ifdef NCCL
            ncclUniqueId NCCLid_;
            ncclComm_t nccl_comm_;
    #endif
#endif
    };

}
