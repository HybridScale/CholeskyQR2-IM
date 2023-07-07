#include "cholesky_qr.hpp"
#include "cqr2bgs.hpp"

cqr::qr2bgs::qr2bgs(std::int64_t m, std::int64_t n, std::int64_t panel_size) : 
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
    cudaI_.resize(n_*n_);

    cudatmp_.resize(input_panel_size_*input_panel_size_);
    cudatmp_.memset(0);
    cudaWtmp_.resize(m_*n_);
    cudaWtmp_.memset(0);

}


cqr::qr2bgs::~qr2bgs()
{
    cublasDestroy(cublashandle_);
    cusolverDnDestroy(cusolverhandle_);

#ifdef NCCL
        NCCLCHECK(ncclCommDestroy(nccl_comm_));
#endif

    MPI_Finalize();
}


void cqr::qr2bgs::InputMatrix(std::vector<double> &A)
{    
}

void cqr::qr2bgs::InputMatrix(double *A)
{
}


void cqr::qr2bgs::InputMatrix(std::string filename)
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


void cqr::qr2bgs::Start()
{

    cudaAlocal_.copytodevice(Alocal_);

    MPI_Warmup();
    MPI_Barrier(MPI_COMM_WORLD);

    timing->start_timing("algorithm");

    cqr2bgs(cudaAlocal_, cudaR_);
    
    timing->stop_timing("algorithm");

    cudaDeviceSynchronize();
    std::vector<int> displacements = distmatrix->get_displacements();
    std::vector<int> counts = distmatrix->get_counts();

    MPI_Gatherv(cudaAlocal_.data(), 
                localm_, distmatrix->get_datatype(), 
                A_.data(),
                counts.data(),
                displacements.data(), 
                distmatrix->get_datatype(), 0, mpi_comm_);

    if( world_rank_ == 0)
    {   
        cudaR_.copytohost(R_);
        validate = std::make_unique<Validate>(m_, n_,
                                                 A_.data(), 
                                                 R_.data(),
                                                 filename_.data());
        orthogonality_ = validate->orthogonality();
        residuals_     = validate->residuals();
        std::cout << "orthogonality: " << orthogonality_ << std::endl;
        std::cout << "residuals: "     << residuals_     << std::endl;
        timing->print();
    }

}


void cqr::qr2bgs::cqr2bgs(cudamemory<double> &A, cudamemory<double> &R)
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
        cqrbgs(A, cudaR2_);
    }

    {
        timing->start_timing("computation");
        NvtxTracer T("R2R1");
        CUBLAS_CHECK(cublasDgemm(cublashandle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                n_,  n_,  n_,
                                &alpha,
                                cudaR1_.data(), n_,
                                cudaR2_.data(), n_,
                                &beta, cudaR_.data(), n_));
        timing->stop_timing("computation");
    }
}


void cqr::qr2bgs::cqrbgs(cudamemory<double> &A, cudamemory<double> &R)
{   
    panel_size_ = input_panel_size_;
    int number_of_panels = ceil((double)n_ / panel_size_);
    for(int j = 0; j < number_of_panels; ++j )
    {   
        panel_size_ = (j == number_of_panels-1) ? n_ - (j) * panel_size_: panel_size_;

        {
            cudatmp_.memset();
            NvtxTracer T("gram");
            gramMatrix(A.data() + j*input_panel_size_, 
                       R.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                       cudatmp_.data());
        }

        timing->start_timing("computation");
        {
            NvtxTracer T("chol");
            cholesky(R.data() + j*input_panel_size_*n_ + j*input_panel_size_);
        }
        {
            NvtxTracer T("Q");
            calculateQ(A.data() + j*input_panel_size_, R.data() + j*input_panel_size_*n_ + j*input_panel_size_);
        }
        timing->stop_timing("computation");

        if(j < number_of_panels - 1 )
        {
            NvtxTracer T("update");
            updateMatrix(n_-(j+1) * input_panel_size_, n_ - input_panel_size_,
                         A.data() + j * input_panel_size_,
                         R.data() + j * input_panel_size_* n_ + j * input_panel_size_ + input_panel_size_);
        }
    }
}


void cqr::qr2bgs::gramMatrix(double *A, double *R, double *tmp) 
{
    // Calculating partial gram matrix to tmp device memory 
    // Sumation of all partial gramm matrix with mpi/nccl allreduce call
    // gemm operation to save to whole R matrix
    double alpha = 1.0, beta = 0.0;

    int n = panel_size_, k = localm_;
    int lda = n_ , ldtmp = panel_size_;
    int ldi = n_, ldr = n_;
    
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDsyrk(cublashandle_,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha,
                             A, lda,
                             &beta,
                             tmp, ldtmp));
   timing->stop_timing("computation");


    timing->start_timing("communication");
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(tmp, tmp, n*n, ncclDouble, ncclSum, nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, tmp, panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    timing->stop_timing("communication");

    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             n, n,
                             &alpha,
                             cudaI_.data(), ldi,
                             tmp, ldtmp,
                             R, ldr));
    timing->stop_timing("computation");

}


void cqr::qr2bgs::cholesky(double *B)
{
    
    size_t d_lwork = 0;     
    size_t h_lwork = 0;     
    void *h_work = nullptr; 

    std::int64_t n = panel_size_, lda = n_;

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


void cqr::qr2bgs::calculateQ(double *A, double *R)
{
    double alpha = 1.0;
    std::int64_t ldr = n_, lda = n_;
    //calculate Q by solving QR = A
    cublasDtrsm(cublashandle_,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                panel_size_, localm_, &alpha, R, ldr, A, lda);

}


void cqr::qr2bgs::updateMatrix(int n, int ldw, double *A, double *R)
{
    double alpha = 1.0, beta=0.0;

    int m = localm_;
    int lda = n_, ldr = n_;

    // W = Qj^T @ Ar
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             n, panel_size_, m,
                             &alpha,
                             A+panel_size_, lda,
                             A, lda,
                             &beta,
                             cudaWtmp_.data(), ldw));
    timing->stop_timing("computation");
    
    timing->start_timing("communication");
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(cudaWtmp_.data(), cudaWtmp_.data(), ldw*panel_size_, ncclDouble, ncclSum,  nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), ldw*panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    #endif
    timing->stop_timing("communication");

    alpha = -1.0;
    beta = 1.0;

    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, panel_size_,
                             &alpha,
                             cudaWtmp_.data(), ldw,
                             A, lda,
                             &beta,
                             A+panel_size_, lda));

    alpha = 1.0;
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             n, panel_size_,
                             &alpha,
                             cudaI_.data(), lda,
                             cudaWtmp_.data(), ldw,
                             R, ldr));
    timing->stop_timing("computation");
}


void cqr::qr2bgs::MPI_Warmup()
{
#ifdef GPU
    MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), m_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
#else
    //MPI_Allreduce(MPI_IN_PLACE, cudaWtmp1_.data(), m_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
#endif

}