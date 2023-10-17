#include <string>

#include "cholesky_qr.hpp"
#include "cqr2bgs.hpp"
#include "gschol.hpp"

cqr::gschol::gschol(std::int64_t m, std::int64_t n, std::int64_t panel_size) : 
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
    cublasSetMathMode(cublashandle_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION );
    
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverhandle_));

    cudaAlocal_.resize(localm_*n_);
    cudaR_.resize(n_*n_);
    //cudaR1_.resize(n_*n_);
    //cudaR2_.resize(n_*n_);
    cudaI_.resize(1);

    //cudatmp_.resize(input_panel_size_*input_panel_size_);
    cudatmp_.resize(input_panel_size_*input_panel_size_);
    cudatmp_.memset(0);
    cudaWtmp_.resize(localm_* (n_ -input_panel_size_));
    cudaWtmp_.memset(0);

}


cqr::gschol::~gschol()
{
    cublasDestroy(cublashandle_);
    cusolverDnDestroy(cusolverhandle_);

#ifdef NCCL
        NCCLCHECK(ncclCommDestroy(nccl_comm_));
#endif

    MPI_Finalize();
}


void cqr::gschol::InputMatrix(std::vector<double> &A)
{    
}

void cqr::gschol::InputMatrix(double *A)
{
}


void cqr::gschol::InputMatrix(std::string filename)
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


void cqr::gschol::Start()
{

    cudaAlocal_.copytodevice(Alocal_);

    MPI_Warmup();
    MPI_Barrier(MPI_COMM_WORLD);

    timing->start_timing("algorithm");

    gschol2(cudaAlocal_, cudaR_);
    
    timing->stop_timing("algorithm");

    cudaDeviceSynchronize();
    std::vector<int> displacements = distmatrix->get_displacements();
    std::vector<int> counts = distmatrix->get_counts();

    cudaAlocal_.copytohost(Alocal_);

    MPI_Gatherv(Alocal_.data(),//cudaAlocal_.data(), 
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


void cqr::gschol::gschol2(cudamemory<double> &A, cudamemory<double> &R)
{
    double alpha = 1.0, beta = 0.0;
    // by panels
    panel_size_ = input_panel_size_;

    int number_of_panels = ceil((double)n_ / panel_size_);

    first_panel_orth();
    std::string name = std::string("Q") + std::to_string(world_rank_) + std::string("_") + std::string(".bin");
    //A.savematrix(name.data()); 

    for(int j = 1; j < number_of_panels; ++j )
    {   
        panel_size_ = (j == number_of_panels-1) ? n_ - (j) * panel_size_: panel_size_;

        //if(j < number_of_panels - 1 )
        {
            NvtxTracer T("update");
            cudaWtmp_.memset();
            //std::string name = std::string("Qub") + std::to_string(world_rank_) + std::string("_")  + std::string(".bin");
            //A.savematrix(name.data());  
            update_rest_Matrix(A, j);
            save_R(cudaR_.data() + (j-1)*input_panel_size_*n_ + j*input_panel_size_,
                   n_, cudaWtmp_.data(), n_ - j * input_panel_size_, input_panel_size_, n_ - j * input_panel_size_);
            //std::string name = std::string("R") + std::to_string(world_rank_) + std::string("_")  + std::string(".bin");
            //cudaR_.savematrix(name.data());  
        }

        {
            cudatmp_.memset();
            NvtxTracer T("gram");
            //std::string name = std::string("Qcb") + std::to_string(world_rank_) + std::string("_") + std::string(".bin");
            //A.savematrix(name.data());
            gramMatrix(A.data() + j*input_panel_size_, 
                       cudaR_.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                       cudatmp_.data());
            cholesky(cudatmp_.data());
            calculateQ(A.data() + j*input_panel_size_, 
                       cudatmp_.data());
            //save_triangular_R(cudaR_.data() + j*input_panel_size_*n_ + j*input_panel_size_,
            //       n_, cudatmp_.data(), panel_size_, panel_size_, panel_size_);
            save_R(cudaR_.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                   n_, cudatmp_.data(), panel_size_, panel_size_, panel_size_);
            //save_R(cudaR_.data() + j*input_panel_size_*n_ + j*input_panel_size_, cudatmp_.data());
            //name = std::string("Qca") + std::to_string(world_rank_) + std::string("_") + std::string(".bin");
            //A.savematrix(name.data());
        }

        //if(j < number_of_panels - 1 )
        {
            NvtxTracer T("update");
            //std::string name = std::string("Qrb") + std::to_string(world_rank_) + std::string("_") + std::to_string(j) + std::string(".bin");
            //A.savematrix(name.data());
            reothrogonalize_panel(A, j);
            //name = std::string("Qra") + std::to_string(world_rank_) + std::string("_") + std::to_string(j) + std::string(".bin");
            //A.savematrix(name.data());
        }

        {
            cudatmp_.memset();
            NvtxTracer T("gram");
            //std::string name = std::string("Qcb") + std::to_string(world_rank_) + std::string("_") + std::string(".bin");
            //A.savematrix(name.data());
            gramMatrix(A.data() + j*input_panel_size_, 
                       cudaR2_.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                       cudatmp_.data());
            cholesky(cudatmp_.data());
            calculateQ(A.data() + j*input_panel_size_, 
                       cudatmp_.data());
            multiply_R(cudaR_.data() + j*input_panel_size_*n_ + j*input_panel_size_, cudatmp_.data());
 
        }
    }
    //A.savematrix("Q0.bin");
    /*
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
    */
   //cudaR_.savematrix("R.bin");
   //A.savematrix("Q.bin");
}


void cqr::gschol::first_panel_orth()
{
    {
        cudatmp_.memset();
        NvtxTracer T("choleksy1");
        gramMatrix(cudaAlocal_.data(), 
                   cudaR_.data(),
                   cudatmp_.data());
        // saving tmp to R
        //cudatmp_.savematrix("tmp.bin");
        cholesky(cudatmp_.data());
        //cudatmp_.savematrix("chol.bin");
        calculateQ(cudaAlocal_.data(), 
                   cudatmp_.data());
        save_R(cudaR_.data(), n_, cudatmp_.data(), panel_size_, panel_size_, panel_size_);
        //save_triangular_R(cudaR_.data(), n_, cudatmp_.data(), panel_size_, panel_size_, panel_size_);

    }
    {
        cudatmp_.memset();
        NvtxTracer T("choleksy2");
        gramMatrix(cudaAlocal_.data(), 
                   cudaR2_.data(),
                   cudatmp_.data());

        cholesky(cudatmp_.data());
        calculateQ(cudaAlocal_.data(), 
                   cudatmp_.data());
        multiply_R(cudaR_.data(), cudatmp_.data());
    }
}

void cqr::gschol::gramMatrix(double *A, double *R, double *tmp) 
{
    // Calculating partial gram matrix to tmp device memory 
    // Sumation of all partial gramm matrix with mpi/nccl allreduce call
    // gemm operation to save to whole R matrix
    double alpha = 1.0, beta = 0.0;

    int n = panel_size_, k = localm_;
    int lda = n_ , ldtmp = panel_size_;
    int ldi = n_, ldr = n_;
    
    timing->start_timing("computation");
    ///*
    CUBLAS_CHECK(cublasDsyrk(cublashandle_,
                             CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N,
                             n, k,
                             &alpha,
                             A, lda,
                             &beta,
                             tmp, ldtmp));
    //*/
    /*
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            n,  n,  localm_,
                            &alpha,
                            A, n_,
                            A, n_,
                            &beta, tmp, ldtmp));
    */
    timing->stop_timing("computation");

    timing->start_timing("communication");
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(tmp, tmp, n*n, ncclDouble, ncclSum, nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, tmp, panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif
    timing->stop_timing("communication");
}


void cqr::gschol::cholesky(double *B)
{
    
    size_t d_lwork = 0;     
    size_t h_lwork = 0;     
    void *h_work = nullptr; 

    std::int64_t n = panel_size_, lda = panel_size_;
    
    timing->start_timing("computation");
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
    timing->stop_timing("computation");
}


void cqr::gschol::calculateQ(double *A, double *R)
{
    timing->start_timing("computation");
    double alpha = 1.0;
    std::int64_t ldr = panel_size_, lda = n_;
    //calculate Q by solving QR = A
    cublasDtrsm(cublashandle_,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                panel_size_, localm_, &alpha, R, ldr, A, lda);
    timing->stop_timing("computation");
}


void cqr::gschol::updateMatrix(int n, int ldw, double *A, double *R)
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

void cqr::gschol::save_R(double* R, std::size_t ldr, double* tmp, std::size_t ldtmp, int m, int n)
{
    timing->start_timing("computation");
    cudaMemcpy2D(R, sizeof(double) * ldr, tmp, sizeof(double) * ldtmp, sizeof(double) * n, m, cudaMemcpyDeviceToDevice);
    timing->stop_timing("computation");
}

void cqr::gschol::save_triangular_R(double *R, std::size_t ldr, double *tmp, std::size_t ldtmp, int m, int n)
{
    double alpha = 1.0;
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                             CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                             n, m,
                             &alpha,
                             cudaI_.data(), n,
                             tmp, ldtmp,
                             R, ldr));

}

void cqr::gschol::multiply_R(double *R, double *tmp)
{
    double alpha = 1.0;
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDtrmm(cublashandle_,
                            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            panel_size_, panel_size_,
                            &alpha,
                            R, n_,
                            tmp, panel_size_,
                            R, n_));
    timing->stop_timing("computation");


}

void cqr::gschol::reothrogonalize_panel(cudamemory<double> &A, int panel_number)
{
    double alpha = 1.0, beta=0.0;

    int m = localm_;
    int lda = n_, ldr = n_;

    // W = Qj^T @ Ar
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             panel_size_, panel_number * input_panel_size_, localm_,
                             &alpha,
                             A.data() + panel_number * input_panel_size_, lda,
                             A.data(), lda,
                             &beta,
                             cudaWtmp_.data(), panel_size_));
    timing->stop_timing("computation");
    
    timing->start_timing("communication");
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(cudaWtmp_.data(), cudaWtmp_.data(), panel_number * input_panel_size_ * panel_size_, ncclDouble, ncclSum,  nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), panel_number * input_panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    #endif
    timing->stop_timing("communication");

    alpha = -1.0;
    beta = 1.0;

    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             panel_size_, localm_, panel_number * input_panel_size_,
                             &alpha,
                             cudaWtmp_.data(), panel_size_,
                             A.data(), lda,
                             &beta,
                             A.data() + panel_number * input_panel_size_, lda));
    timing->stop_timing("computation");

}

void cqr::gschol::update_rest_Matrix(cudamemory<double> &A, int panel_number)
{
    double alpha = 1.0, beta=0.0;

    int m = localm_;
    int lda = n_, ldr = n_;
    int ldw = n_ - panel_number * input_panel_size_;

    // W = Qj^T @ Ar
    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_T,
                             n_ - panel_number * input_panel_size_, input_panel_size_, localm_,
                             &alpha,
                             A.data() + panel_number * input_panel_size_, lda,
                             A.data() + (panel_number -1) * input_panel_size_, lda,
                             &beta,
                             cudaWtmp_.data(), ldw));
    timing->stop_timing("computation");
    
    timing->start_timing("communication");
    #ifdef NCCL
        NCCLCHECK(ncclAllReduce(cudaWtmp_.data(), cudaWtmp_.data(), ldw * input_panel_size_, ncclDouble, ncclSum,  nccl_comm_, 0));
    #else
        MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), ldw * input_panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    #endif
    timing->stop_timing("communication");

    alpha = -1.0;
    beta = 1.0;

    timing->start_timing("computation");
    CUBLAS_CHECK(cublasDgemm(cublashandle_,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n_ - panel_number * input_panel_size_, localm_, input_panel_size_,
                             &alpha,
                             cudaWtmp_.data(), ldw,
                             A.data() + (panel_number -1) * input_panel_size_, lda,
                             &beta,
                             A.data() + panel_number * input_panel_size_, lda));
    timing->stop_timing("computation");
}

void cqr::gschol::MPI_Warmup()
{
#ifdef GPU
    if (localm_ >= n_)
        MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), n_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    else
        MPI_Allreduce(MPI_IN_PLACE, cudaWtmp_.data(), 10 * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
#else
    //MPI_Allreduce(MPI_IN_PLACE, cudaWtmp1_.data(), n_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
#endif
}
