/*
 * File:   cqr2bgslookahead.cpp
 * Date:   July 7, 2023
 * Brief:  Implementation of the CholeskyQR2 with modified block Gram-Schmidt reorthogonalization algorithm,
 *         including lookahead optimization. GPU implementation with CUDA-aware MPI or NCCL communicators.
 * 
 * This file is part of the CholeskyQR2++ library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */

#include "cqr2bgslookahead.hpp"

cqr::qr2bgsloohahead::qr2bgsloohahead(std::int64_t m, std::int64_t n, std::int64_t panel_size) : 
                                      m_(m), n_(n), input_panel_size_(panel_size)
{
    
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm_);
    MPI_Comm_size(shmcomm_, &shmsize_);
    MPI_Comm_rank(shmcomm_, &shmrank_);   

    mpi_comm_ = MPI_COMM_WORLD;

    distmatrix = std::make_unique<DistributedMatrix>(m_, n_, mpi_comm_);
    timing = std::make_unique<TimingGpu>();

    localm_    = distmatrix->get_rank_localm();

    if (world_rank_ == 0)
    {
        size = m_ * n_;
        A_.resize( size);
    }

    Alocal_.resize( localm_ * n_); 
    R_.resize( n_ * n_);


    //CUDA_CHECK(cudaSetDevice ( shmrank_ ));
    cudaSetDevice ( shmrank_ );
    
#ifdef NCCL
        if (world_rank_ == 0) 
            NCCLCHECK(ncclGetUniqueId(&NCCLid_));

        MPI_Bcast(&NCCLid_, sizeof(NCCLid_), MPI_BYTE, 0, mpi_comm_);
        NCCLCHECK(ncclCommInitRank(&nccl_comm_, world_size_, NCCLid_, world_rank_));
#endif
    

    int leastPriority, greatestPriority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange( &leastPriority, &greatestPriority )); 
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_panelqr1, cudaStreamDefault, greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_panelqr2, cudaStreamDefault, greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_update1, cudaStreamDefault, greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_update2, cudaStreamDefault, leastPriority));

    cudaDeviceGetStreamPriorityRange( &leastPriority, &greatestPriority ); 
    
    cudaStreamCreateWithPriority(&stream_panelqr1, cudaStreamDefault, greatestPriority);
    cudaStreamCreateWithPriority(&stream_panelqr2, cudaStreamDefault, greatestPriority);
    cudaStreamCreateWithPriority(&stream_update1, cudaStreamDefault, greatestPriority);
    cudaStreamCreateWithPriority(&stream_update2, cudaStreamDefault, leastPriority);

    CUBLAS_CHECK(cublasCreate(&cublashandle_));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverhandle_));

    cudaAlocal_.resize(localm_*n_);
    cudaR_.resize(n_*n_);
    cudaR1_.resize(n_*n_);
    cudaR2_.resize(n_*n_);
    cudaI_.resize(n_*n_);

    cudatmp_.resize(input_panel_size_*input_panel_size_);
    cudatmp_.memset();
    cudatmpqr2_.resize(input_panel_size_*input_panel_size_);
    cudatmpqr2_.memset();
    cudaWtmp1_.resize(localm_* (n_ -input_panel_size_));
    cudaWtmp1_.memset();
    cudaWtmp2_.resize(localm_* (n_ -input_panel_size_));
     cudaWtmp2_.memset();

    d_work.resize(n_);
    d_info.resize(1);

}


cqr::qr2bgsloohahead::~qr2bgsloohahead()
{
    cublasDestroy(cublashandle_);
    cusolverDnDestroy(cusolverhandle_);
    
#ifdef NCCL
        NCCLCHECK(ncclCommDestroy(nccl_comm_));
#endif

    MPI_Finalize();
}


void cqr::qr2bgsloohahead::InputMatrix(cudamemory<double> &A)
{
    MPI_File fileHandle;
    MPI_Status status;
    int access_mode = MPI_MODE_RDONLY; // mode for reading only


    if(MPI_File_open(mpi_comm_, filename_.c_str(), access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
    {
        std::cout << "Can't open input matrix - " << filename_ << " on rank " << world_rank_ << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int displacement = distmatrix->get_rank_displacement();
    int counts = distmatrix->get_rank_count();


    MPI_File_read_at_all(fileHandle, displacement * n_* sizeof(double),
                        A.data(), counts,
                        distmatrix->get_datatype(), &status);

    if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
    {
        MPI_Abort(mpi_comm_, EXIT_FAILURE);
    }
}

void cqr::qr2bgsloohahead::InputMatrix(double *A)
{
}


void cqr::qr2bgsloohahead::InputMatrix( const char* filename)
{
    MPI_File fileHandle;
    MPI_Status status;
    int access_mode = MPI_MODE_RDONLY; // mode for reading only

    filename_ = filename;

    if(MPI_File_open(mpi_comm_, filename, access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
    {
        std::cout << "Can't open input matrix - " << filename << " on rank " << world_rank_ << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int displacement = distmatrix->get_rank_displacement();
    int counts = distmatrix->get_rank_count();


    MPI_File_read_at_all(fileHandle, displacement * n_* sizeof(double),
                        Alocal_.data(), counts,
                        distmatrix->get_datatype(), &status);

    if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
    {
        MPI_Abort(mpi_comm_, EXIT_FAILURE);
    }
}


void cqr::qr2bgsloohahead::Start()
{


    cudaAlocal_.copytodevice(Alocal_);

    MPI_Warmup();
    MPI_Barrier(MPI_COMM_WORLD);

    timing->start_timing("algorithm");

    cqr2bgs(cudaAlocal_, cudaR_);

    timing->stop_timing("algorithm");

    cudaR1_.release();
    cudaR2_.release();
    cudaI_.release();

    cudatmp_.release();
    cudatmpqr2_.release();
    cudaWtmp1_.release();
    cudaWtmp2_.release();

    cudaStreamSynchronize(stream_panelqr1);

    std::vector<int> displacements = distmatrix->get_displacements();
    std::vector<int> counts = distmatrix->get_counts();

    /*
    cudaAlocal_.copytohost(Alocal_);

    MPI_Gatherv(Alocal_.data(), 
                localm_, distmatrix->get_datatype(), 
                A_.data(),
                counts.data(),
                displacements.data(), 
                distmatrix->get_datatype(), 0, mpi_comm_);
*/
    validate = std::make_unique<Validate>(localm_, n_,
                                          cudaAlocal_.data(), 
                                          cudaR_.data(),
                                          filename_.c_str(),
                                          cublashandle_);
/*
        cudaR_.copytohost(R_);
        validate = std::make_unique<Validate>(m_, n_,
                                                 A_.data(), 
                                                 R_.data(),
                                                 filename_);
*/
    orthogonality_ = validate->orthogonality();
    cudamemory<double> A(localm_ * n_);
    InputMatrix(A);
    residuals_     = validate->residuals(A);
    if( world_rank_ == 0)
    {  
        std::cout << "orthogonality: " << orthogonality_ << std::endl;
        std::cout << "residuals: "     << residuals_     << std::endl;
        timing->print();
    }

}


void cqr::qr2bgsloohahead::cqr2bgs(cudamemory<double> &A, cudamemory<double> &R)
{
    double alpha = 1.0, beta = 0.0;
    // First cqrbgs call
    {
        NvtxTracer T("CQR1");
        cqrbgs(A, cudaR1_);
    }
    // Second cqrbgs call
    {
        NvtxTracer T("CQR2");
        cudaStreamSynchronize(stream_panelqr1);
        cqrbgs(A, cudaR2_);
    }

    {
        timing->start_timing("computation", stream_panelqr1);
        NvtxTracer T("R2R1");
        CUBLAS_CHECK(cublasDgemm(cublashandle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n_,  n_,  n_,
                                &alpha,
                                cudaR1_.data(), n_,
                                cudaR2_.data(), n_,
                                &beta, cudaR_.data(), n_));
        timing->stop_timing("computation", stream_panelqr1);
    }
}


void cqr::qr2bgsloohahead::cqrbgs(cudamemory<double> &A, cudamemory<double> &R)
{   
    panel_size_ = input_panel_size_;
    int number_of_panels = ceil((double)n_ / panel_size_);
    
    std::int64_t update1_panel_size = input_panel_size_;

    for(int j = 0; j < number_of_panels; ++j )
    {   
        panel_size_ = (j == number_of_panels-1) ? n_ - (j) * input_panel_size_: input_panel_size_;

        {
            NvtxTracer T("gram");
            cudaStreamSynchronize(stream_update1);
            gramMatrixGemm(A.data() + j*input_panel_size_, 
                           R.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                           cudatmp_.data(),
                           stream_panelqr1);
        }

                if(j >0 && j < number_of_panels - 1 )
        {
            NvtxTracer T("update2Rest");
            updateMatrixCommunication(cudaWtmp2_.data(), n_-(j+1) * input_panel_size_,
                            stream_update2);
            updateMatrixRest(localm_, n_-(j+1) * input_panel_size_, panel_size_,
                             A.data() + (j-1) * input_panel_size_, n_,
                             A.data() + (j+1) * input_panel_size_, n_,
                             R.data() + (j-1) * input_panel_size_* n_ + (j-1) * input_panel_size_ +  2* input_panel_size_, n_,
                             cudaWtmp2_.data(), n_-(j+1) * input_panel_size_,
                             stream_update2);
        }

        {
            NvtxTracer T("gramRest");
            gramMatrixCommunication(cudatmp_.data(), stream_panelqr1);
            gramMatrixRest(A.data() + j*input_panel_size_, 
                           R.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                           cudatmp_.data(),
                           stream_panelqr1);
        }
        cudatmp_.memset();
        
        timing->start_timing("computation", stream_panelqr1);
        {
            NvtxTracer T("chol");
            cholesky(R.data() + j*input_panel_size_*n_ + j*input_panel_size_, stream_panelqr1);
        }
        {
            NvtxTracer T("Q");
            calculateQ(A.data() + j*input_panel_size_, R.data() + j*input_panel_size_*n_ + j*input_panel_size_, stream_panelqr1);
        }
        timing->stop_timing("computation", stream_panelqr1);

        //cudaStreamSynchronize(stream_panelqr1);
        
        //cudaStreamSynchronize(stream_update2);

        if(j < number_of_panels - 1 )
        {
            
            NvtxTracer T("update1");
            cudaStreamSynchronize(stream_update2);
            cudaStreamSynchronize(stream_panelqr1);
            if (j == number_of_panels-2)
                update1_panel_size = n_-(j+1) * input_panel_size_;

            updateMatrixGemm(localm_, update1_panel_size, panel_size_,
                             A.data() + j * input_panel_size_, n_,
                             A.data() + (j+1) * input_panel_size_, n_,
                             cudaWtmp1_.data(), update1_panel_size, 
                             stream_update1);
        }

        //cudaStreamSynchronize(stream_update1);

        if(j < number_of_panels - 2 )
        {
            NvtxTracer T("update2");
           updateMatrixGemm(localm_, n_-(j+2) * input_panel_size_, panel_size_,
                            A.data() + j * input_panel_size_, n_,
                            A.data() + (j+2) * input_panel_size_, n_,
                            cudaWtmp2_.data(), n_-(j+2) * input_panel_size_,
                            stream_update2);
        }

        //cudaStreamSynchronize(stream_update2);

        if(j < number_of_panels - 1 )
        {
            NvtxTracer T("update1Rest");
            updateMatrixCommunication(cudaWtmp1_.data(), update1_panel_size, 
                                      stream_update1);
            updateMatrixRest(localm_, update1_panel_size, panel_size_,
                         A.data() + j * input_panel_size_, n_,
                         A.data() + (j+1) * input_panel_size_, n_,
                         R.data() + j * input_panel_size_* n_ + j * input_panel_size_ + input_panel_size_, n_,
                         cudaWtmp1_.data(), update1_panel_size, 
                         stream_update1);
        }
        //cudaStreamSynchronize(stream_update1);


/*
        if( j < number_of_panels - 2 )
        {
            NvtxTracer T("update2Rest");
            updateMatrixCommunication(cudaWtmp2_.data(), n_-(j+2) * input_panel_size_,
                            stream_update2);
            updateMatrixRest(localm_, n_-(j+2) * input_panel_size_, panel_size_,
                             A.data() + (j) * input_panel_size_, n_,
                             A.data() + (j+2) * input_panel_size_, n_,
                             R.data() + (j) * input_panel_size_* n_ + (j) * input_panel_size_ +  2* input_panel_size_, n_,
                             cudaWtmp2_.data(), n_-(j+2) * input_panel_size_,
                             stream_update2);
        }
*/
        //cudaStreamSynchronize(stream_update2);

    }
}


void cqr::qr2bgsloohahead::gramMatrixGemm(double *A, double *R, double *tmp, cudaStream_t stream)
{
    double alpha = 1.0, beta = 0.0;

    int n = panel_size_, k = localm_;
    int lda = n_ , ldtmp = panel_size_;
    int ldi = n_, ldr = n_;

    CUBLAS_CHECK(cublasSetStream(cublashandle_, stream));
    
    timing->start_timing("computation", stream);
    CUBLAS_CHECK(cublasDsyrk(cublashandle_,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha,
                             A, lda,
                             &beta,
                             tmp, ldtmp));
   timing->stop_timing("computation", stream);

}


void cqr::qr2bgsloohahead::gramMatrixCommunication(double *tmp, cudaStream_t stream)
{
    timing->start_timing("communication", stream);
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(tmp, tmp, panel_size_ * panel_size_, ncclDouble, ncclSum, nccl_comm_, stream));
    #else
        cudaStreamSynchronize(stream);
        MPI_Allreduce(MPI_IN_PLACE, tmp, panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    timing->stop_timing("communication", stream);
}


void cqr::qr2bgsloohahead::gramMatrixRest(double *A, double *R, double *tmp, cudaStream_t stream)
{
    double alpha = 1.0, beta = 0.0;

    int n = panel_size_, k = localm_;
    int lda = n_ , ldtmp = panel_size_;
    int ldi = n_, ldr = n_;

    CUBLAS_CHECK(cublasSetStream(cublashandle_, stream));
    timing->start_timing("computation", stream);
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             n, n,
                             &alpha,
                             cudaI_.data(), ldi,
                             tmp, ldtmp,
                             R, ldr));
    timing->stop_timing("computation", stream);
}


void cqr::qr2bgsloohahead::cholesky(double *B, cudaStream_t stream)
{
    
    size_t d_lwork = 0;     
    size_t h_lwork = 0;     
    void *h_work = nullptr; 

    std::int64_t n = panel_size_, lda = n_;

    CUSOLVER_CHECK(cusolverDnSetStream(cusolverhandle_, stream));

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverhandle_, NULL, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, B, lda,
        CUDA_R_64F, &d_lwork, &h_lwork));

    // Allocate buffer for holding scratchpad memory to be used by the routine 
    // for storing intermediate results

    CUSOLVER_CHECK(cusolverDnXpotrf(cusolverhandle_, NULL, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F,
        B, lda, CUDA_R_64F, d_work.data(), d_lwork,
        h_work, h_lwork, d_info.data()));
}


void cqr::qr2bgsloohahead::calculateQ(double *A, double *R, cudaStream_t stream)
{
    double alpha = 1.0;
    std::int64_t ldr = n_, lda = n_;
    
    CUBLAS_CHECK(cublasSetStream(cublashandle_, stream));

    //calculate Q by solving QR = A
    cublasDtrsm(cublashandle_,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                panel_size_, localm_, &alpha, R, ldr, A, lda);

}


void cqr::qr2bgsloohahead::updateMatrixGemm(std::int64_t m, std::int64_t n, std::int64_t k, 
                                            double *A, std::int64_t lda,
                                            double *B, std::int64_t ldb,
                                            double *Tmp, std::int64_t ldw,
                                            cudaStream_t stream)
{
    double alpha = 1.0, beta=0.0;

    CUBLAS_CHECK(cublasSetStream(cublashandle_, stream));

    timing->start_timing("computation", stream);
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             n, k, m,
                             &alpha,
                             B, lda,
                             A, lda,
                             &beta,
                             Tmp, ldw));
    timing->stop_timing("computation", stream);
}


void cqr::qr2bgsloohahead::updateMatrixCommunication( double *Tmp, std::int64_t ldw, cudaStream_t stream)
{
    timing->start_timing("communication", stream);
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(Tmp, Tmp, ldw*panel_size_, ncclDouble, ncclSum,  nccl_comm_, stream));
    #else
        cudaStreamSynchronize(stream);
        
        MPI_Allreduce(MPI_IN_PLACE, Tmp, ldw*panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
        //MPI_Request test;
        //MPI_Iallreduce(MPI_IN_PLACE, Tmp, ldw*panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_, &test);
    #endif
    timing->stop_timing("communication", stream);
}


void cqr::qr2bgsloohahead::updateMatrixRest(std::int64_t m, std::int64_t n, std::int64_t k, 
                                            double *A, std::int64_t lda, 
                                            double *B, std::int64_t ldb, 
                                            double *C, std::int64_t ldc, 
                                            double *Tmp, std::int64_t ldw, 
                                            cudaStream_t stream)
{
    double alpha = -1.0;
    double beta = 1.0;

    CUBLAS_CHECK(cublasSetStream(cublashandle_, stream));

    timing->start_timing("computation", stream);
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             Tmp, ldw,
                             A, lda,
                             &beta,
                             B, ldb));

    alpha = 1.0;
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             n, k,
                             &alpha,
                             cudaI_.data(), lda,
                             Tmp, ldw,
                             C, ldc));
    timing->stop_timing("computation", stream);
}

void cqr::qr2bgsloohahead::MPI_Warmup()
{
#ifdef GPU 
    MPI_Allreduce(MPI_IN_PLACE, cudaWtmp1_.data(), (n_ -input_panel_size_), MPI_DOUBLE, MPI_SUM, mpi_comm_);
#else
   //MPI_Allreduce(MPI_IN_PLACE, cudaWtmp1_.data(), n_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
#endif

}