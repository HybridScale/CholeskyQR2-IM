#include "scqr3.hpp"

cqr::qr3::qr3(std::int64_t m, std::int64_t n) : 
                 m_(m), n_(n)
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

    CUDA_CHECK(cudaSetDevice ( shmrank_ ));

#ifdef NCCL
    if (world_rank_ == 0) 
        NCCLCHECK(ncclGetUniqueId(&NCCLid_));

    MPI_Bcast(&NCCLid_, sizeof(NCCLid_), MPI_BYTE, 0, mpi_comm_);
    NCCLCHECK(ncclCommInitRank(&nccl_comm_, world_size_, NCCLid_, world_rank_));
#endif

    CUBLAS_CHECK(cublasCreate(&cublashandle_));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverhandle_));

    cudaAlocal_.resize(localm_*n_);
    cudaR_.resize(n_*n_);
    cudaR1_.resize(n_*n_);
    cudaR2_.resize(n_*n_);
}


cqr::qr3::~qr3()
{
    cublasDestroy(cublashandle_);
    cusolverDnDestroy(cusolverhandle_);

#ifdef NCCL
        NCCLCHECK(ncclCommDestroy(nccl_comm_));
#endif

    MPI_Finalize();
}


void cqr::qr3::InputMatrix(cudamemory<double> &A)
{
    MPI_File fileHandle;
    MPI_Status status;
    int access_mode = MPI_MODE_RDONLY; // mode for reading only


    if(MPI_File_open(mpi_comm_, filename_.data(), access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
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

void cqr::qr3::InputMatrix(double *A)
{
}

void cqr::qr3::InputMatrix(std::string filename)
{

    MPI_File fileHandle;
    MPI_Status status;
    int access_mode = MPI_MODE_RDONLY; // mode for reading only

    filename_ = filename;

    if(MPI_File_open(mpi_comm_, filename.data(), access_mode, MPI_INFO_NULL, &fileHandle) != MPI_SUCCESS)
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


void cqr::qr3::Start()
{

    cudaAlocal_.copytodevice(Alocal_);

    MPI_Warmup();
    MPI_Barrier(MPI_COMM_WORLD);

    timing->start_timing("algorithm");

    scqr3(cudaAlocal_, cudaR_);
    
    timing->stop_timing("algorithm");

    cudaR1_.release();
    cudaR2_.release();

    cudaDeviceSynchronize();
    std::vector<int> displacements = distmatrix->get_displacements();
    std::vector<int> counts = distmatrix->get_counts();
 
    validate = std::make_unique<Validate>(localm_, n_,
                                            cudaAlocal_.data(), 
                                            cudaR_.data(),
                                            filename_.data(),
                                            cublashandle_);
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


void cqr::qr3::scqr3(cudamemory<double> &A, cudamemory<double> &R)
{
    double alpha = 1.0, beta = 0.0;

    //Obtaining Frobenius norm of input matrix:
    {
        FrobeniusNorm(A.data());
    }
    shift=pow(frnorm,2)*sqrt(m_)*1e-16 * 1e-3;
    
    //First call: ShiftedCholeskyQR
    {
        NvtxTracer T("SCQR");
        scqr(A, cudaR_);
    }  

    //Second call: CholeskyQR2
    {
        NvtxTracer T("CQR2");
        cqr(A, cudaR1_);

        double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublashandle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n_,  n_,  n_,
                                &alpha,
                                cudaR_.data(), n_,
                                cudaR1_.data(), n_,
                                &beta, cudaR2_.data(), n_));
        cqr(A, cudaR1_);
        CUBLAS_CHECK(cublasDgemm(cublashandle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n_,  n_,  n_,
                                &alpha,
                                cudaR2_.data(), n_,
                                cudaR1_.data(), n_,
                                &beta, cudaR_.data(), n_));
    }
}


void cqr::qr3::cqr(cudamemory<double> &A, cudamemory<double> &R)
{   
    {
        NvtxTracer T("gram");
        gramMatrix(A.data(), R.data());
    }

    timing->start_timing("computation");
    {
        NvtxTracer T("chol");
        cholesky(R.data());
    }
    {
        NvtxTracer T("Q");
        calculateQ(A.data(), R.data());
    }
    timing->stop_timing("computation");
}

void cqr::qr3::scqr(cudamemory<double> &A, cudamemory<double> &R)
{
    {
        NvtxTracer T("gram");
        gramMatrixShifted(A.data(), R.data());
    }
    timing->start_timing("computation");
    {
        NvtxTracer T("chol");
        cholesky(R.data());
    }
    {
        NvtxTracer T("Q");
        calculateQ(A.data(), R.data());
    }
    timing->stop_timing("computation");
}

void cqr::qr3::FrobeniusNorm(double *A)
{
    double sqrtofpartialsumofsquares;
    
    std::vector<double> sums(world_size_);
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDnrm2(cublashandle_,n_*localm_, A, 1, &sqrtofpartialsumofsquares));  //Norm function returns the root of the sum over squares. We will need to square this quantity when gathering form all nodes.
    timing->stop_timing("computation");

    timing->start_timing("communication");
    //(annot AK) Allreduce from all nodes to construct final gram matrix:
    MPI_Gather(&sqrtofpartialsumofsquares, 1, MPI_DOUBLE, sums.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    timing->stop_timing("communication");
    if(world_rank_==0)
    {
        frnorm = 0.0;
        for(auto i=0;i<world_size_;++i)
        {
            frnorm += pow(sums[i],2.0);
        }
        frnorm=sqrt(frnorm);
    }
}

void cqr::qr3::gramMatrix(double *A, double *R) 
{
    // Calculating partial gram matrix to tmp device memory 
    // Sumation of all partial gramm matrix with mpi/nccl allreduce call
    // gemm operation to save to whole R matrix
    double alpha = 1.0, beta = 0.0;

    int n = n_, k = localm_;
    int lda = n_ , ldtmp = n_;
    int ldi = n_, ldr = n_;
    
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDsyrk(cublashandle_,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha,
                             A, lda,
                             &beta,
                             R, ldr));
    
   timing->stop_timing("computation");


    timing->start_timing("communication");
    //(annot AK) Allreduce from all nodes to construct final gram matrix:
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(R, R, n_ * n_, ncclDouble, ncclSum, nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, R, n_ * n_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    timing->stop_timing("communication");
}

void cqr::qr3::gramMatrixShifted(double *A, double *R) 
{
    // Calculating partial gram matrix to tmp device memory 
    // (add AK) Shift added to partial gram matrix in one node
    // Sumation of all partial gramm matrix with mpi/nccl allreduce call
    // gemm operation to save to whole R matrix
    double alpha = 1.0, beta = 0.0;

    int n = n_, k = localm_;
    int lda = n_ , ldtmp = n_;
    int ldi = n_, ldr = n_;
    int incx = 1, incy = n_; //(add AK) parameters for Daxpy for shift
    
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDsyrk(cublashandle_,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha,
                             A, lda,
                             &beta,
                             R, ldr));
    
    if( world_rank_ == 0) //(add AK) adding the shift along the diagonal to the partial gram matrix in a single node
    {   
        cudamemory<double> VECones(n);
        VECones.memset(1);
        CUBLAS_CHECK(cublasDaxpy(cublashandle_, n, &shift, VECones.data(), 1, R, n+1))
    }
    
   timing->stop_timing("computation");


    timing->start_timing("communication");
    //(annot AK) Allreduce from all nodes to construct final gram matrix:
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(R, R, n_ * n_, ncclDouble, ncclSum, nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, R, n_ * n_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    timing->stop_timing("communication");
}


void cqr::qr3::cholesky(double *B)
{
    
    size_t d_lwork = 0;     
    size_t h_lwork = 0;     
    void *h_work = nullptr; 

    std::int64_t n = n_, lda = n_;

    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
        cusolverhandle_, NULL, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, B, lda,
        CUDA_R_64F, &d_lwork, &h_lwork));

    // Allocate buffer for holding scratchpad memory to be used by the routine 
    // for storing intermediate results
    cudamemory<double> d_work(d_lwork);
    cudamemory<int> d_info(1);

    CUSOLVER_CHECK(cusolverDnXpotrf(cusolverhandle_, NULL, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F,
        B, lda, CUDA_R_64F, d_work.data(), d_lwork,
        h_work, h_lwork, d_info.data()));
}


void cqr::qr3::calculateQ(double *A, double *R)
{
    double alpha = 1.0;
    std::int64_t ldr = n_, lda = n_;
    //calculate Q by solving QR = A
    cublasDtrsm(cublashandle_,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                n_, localm_, &alpha, R, ldr, A, lda);

}


void cqr::qr3::MPI_Warmup()
{
    MPI_Allreduce(MPI_IN_PLACE, cudaR1_.data(), n_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
}