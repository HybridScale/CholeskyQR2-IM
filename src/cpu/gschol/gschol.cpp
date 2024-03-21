/*
 * File:   gschol.cpp
 * Date:   July 7, 2023
 * Brief:  Implementation of the modified CholeskyQR2 with modified block Gram-Schmidt reorthogonalization algorithm. 
 *         CPU-only implementation with MPI.
 * 
 * This file is part of the CholeskyQR2++ library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */

#include <string>
#include <fstream>
#include <cstring>

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
    timing = std::make_unique<Timing>();

    localm_    = distmatrix->get_rank_localm();

    Alocal_.resize( localm_ * n_); 
    R_.resize( n_ * n_);
    tmp_.resize(input_panel_size_*input_panel_size_);
    Wtmp_.resize(localm_* (n_ -input_panel_size_));
}


cqr::gschol::~gschol()
{
    MPI_Finalize();
}


void cqr::gschol::InputMatrix(double *A)
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
                        A, counts,
                        distmatrix->get_datatype(), &status);

    if (MPI_File_close(&fileHandle) != MPI_SUCCESS)
    {
        MPI_Abort(mpi_comm_, EXIT_FAILURE);
    }
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
    MPI_Warmup();
    MPI_Barrier(MPI_COMM_WORLD);

    timing->start_timing("algorithm");

    gschol2(Alocal_, R_);
    
    timing->stop_timing("algorithm");

    std::vector<int> displacements = distmatrix->get_displacements();
    std::vector<int> counts = distmatrix->get_counts();

    validate = std::make_unique<Validate>(localm_, n_,
                                          Alocal_.data(), 
                                          R_.data(),
                                          filename_.data());
    orthogonality_ = validate->orthogonality();
    std::vector<double> A(localm_ * n_);
    InputMatrix(A.data());
    residuals_     = validate->residuals(A);
    //residuals_     = validate->residuals();
    if( world_rank_ == 0)
    {
        std::cout << "orthogonality: " << orthogonality_ << std::endl;
        std::cout << "residuals: "     << residuals_     << std::endl;
        timing->print();
    }
}


void cqr::gschol::gschol2(std::vector<double> &A, std::vector<double> &R)
{
    double alpha = 1.0, beta = 0.0;
    // by panels
    panel_size_ = input_panel_size_;

    int number_of_panels = ceil((double)n_ / panel_size_);

    first_panel_orth();
    
    for(int j = 1; j < number_of_panels; ++j )
    {   
        panel_size_ = (j == number_of_panels-1) ? n_ - (j) * panel_size_: panel_size_;

        
        update_rest_Matrix(A, j);
        save_R(R_.data() + (j-1)*input_panel_size_*n_ + j*input_panel_size_,
                n_, Wtmp_.data(), n_ - j * input_panel_size_, input_panel_size_, n_ - j * input_panel_size_);

        gramMatrix(A.data() + j*input_panel_size_, 
                    tmp_.data());
        cholesky(tmp_.data());
        calculateQ(A.data() + j*input_panel_size_, 
                    tmp_.data());
        save_R(R_.data() + j*input_panel_size_*n_ + j*input_panel_size_,
                n_, tmp_.data(), panel_size_, panel_size_, panel_size_);
    
        reothrogonalize_panel(A, j);

        gramMatrix(A.data() + j*input_panel_size_, 
                    tmp_.data());
        cholesky(tmp_.data());
        calculateQ(A.data() + j*input_panel_size_, 
                    tmp_.data());
        multiply_R(R_.data() + j*input_panel_size_*n_ + j*input_panel_size_, tmp_.data());
    }
}


void cqr::gschol::first_panel_orth()
{
    
    gramMatrix(Alocal_.data(), 
                tmp_.data());

    cholesky(tmp_.data());
    calculateQ(Alocal_.data(), 
               tmp_.data());

    save_R(R_.data(), n_, tmp_.data(), panel_size_, panel_size_, panel_size_);

    gramMatrix(Alocal_.data(), 
                tmp_.data());

    cholesky(tmp_.data());
    calculateQ(Alocal_.data(), 
                tmp_.data());
    multiply_R(R_.data(), tmp_.data());
}

void cqr::gschol::gramMatrix(double *A, double *tmp) 
{
    // Calculating partial gram matrix to tmp device memory 
    // Sumation of all partial gramm matrix with mpi/nccl allreduce call
    // gemm operation to save to whole R matrix
    double alpha = 1.0, beta = 0.0;

    std::int64_t n = panel_size_, k = localm_;
    std::int64_t lda = n_ , ldtmp = panel_size_;
    std::int64_t ldi = n_, ldr = n_;
    
    timing->start_timing("computation");
    dsyrk("L", "N",
          &n, &k,
          &alpha,
          A, &lda,
          &beta,
          tmp, &ldtmp);
    timing->stop_timing("computation");

    timing->start_timing("communication");
    MPI_Allreduce(MPI_IN_PLACE, tmp, panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    timing->stop_timing("communication");
}


void cqr::gschol::cholesky(double *B)
{
    std::int64_t n = panel_size_, lda = panel_size_;
    timing->start_timing("computation");

    std::int64_t info = 0;
    dpotrf("L", &n, B, &lda, &info);
    timing->stop_timing("computation");
}


void cqr::gschol::calculateQ(double *A, double *R)
{
    timing->start_timing("computation");
    double alpha = 1.0;
    std::int64_t ldr = panel_size_, lda = n_;
    //calculate Q by solving QR = A
    dtrsm("L", "L", "N", "N",
          &panel_size_, &localm_, &alpha, R, &ldr, A, &lda);
    timing->stop_timing("computation");
}


void cqr::gschol::save_R(double* R, std::int64_t ldr, double* tmp, std::int64_t ldtmp, std::int64_t m, std::int64_t n)
{
    timing->start_timing("computation");
    dlacpy("N", &n, &m, tmp, &ldtmp, R, &ldr );
    timing->stop_timing("computation");
}

void cqr::gschol::multiply_R(double *R, double *tmp)
{
    double alpha = 1.0;
    timing->start_timing("computation");

    dtrmm("R", "L", "N", "N", 
          &panel_size_, &panel_size_,
          &alpha,
          tmp, &panel_size_,
          R, &n_);
    timing->stop_timing("computation");
}


void cqr::gschol::reothrogonalize_panel(std::vector<double> &A, int panel_number)
{
    double alpha = 1.0, beta=0.0;

    std::int64_t m = localm_, n = panel_number * input_panel_size_;
    std::int64_t lda = n_, ldr = n_;

    // W = Qj^T @ Ar
    timing->start_timing("computation");

    dgemm("N", "T",
          &panel_size_, &n, &localm_,
          &alpha,
          A.data() + panel_number * input_panel_size_, &lda,
          A.data(), &lda,
          &beta,
          Wtmp_.data(), &panel_size_);

    timing->stop_timing("computation");
    
    timing->start_timing("communication");
    MPI_Allreduce(MPI_IN_PLACE, Wtmp_.data(), panel_number * input_panel_size_ * panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    timing->stop_timing("communication");

    alpha = -1.0;
    beta = 1.0;

    timing->start_timing("computation");

    dgemm("N", "N",
          &panel_size_, &localm_, &n,
          &alpha,
          Wtmp_.data(), &panel_size_,
          A.data(), &lda,
          &beta,
          A.data() + panel_number * input_panel_size_, &lda);
    timing->stop_timing("computation");

}

void cqr::gschol::update_rest_Matrix(std::vector<double> &A, int panel_number)
{
    double alpha = 1.0, beta=0.0;

    std::int64_t m = n_ - panel_number * input_panel_size_;
    std::int64_t lda = n_, ldr = n_;
    std::int64_t ldw = n_ - panel_number * input_panel_size_;

    // W = Qj^T @ Ar
    timing->start_timing("computation");
    dgemm("N", "T", 
          &m, &input_panel_size_, &localm_, 
          &alpha, 
          A.data() + panel_number * input_panel_size_, &lda,
          A.data() + (panel_number -1) * input_panel_size_, &lda,
          &beta,
          Wtmp_.data(), &ldw);
    timing->stop_timing("computation");
    
    timing->start_timing("communication");
    MPI_Allreduce(MPI_IN_PLACE, Wtmp_.data(), ldw * input_panel_size_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
    timing->stop_timing("communication");

    alpha = -1.0;
    beta = 1.0;

    timing->start_timing("computation");
    dgemm("N", "N",
          &m, &localm_, &input_panel_size_, 
          &alpha, 
          Wtmp_.data(), &ldw,
          A.data() + (panel_number -1) * input_panel_size_, &lda,
          &beta,
          A.data() + panel_number * input_panel_size_, &lda);
    timing->stop_timing("computation");
}


void cqr::gschol::MPI_Warmup()
{
    MPI_Allreduce(MPI_IN_PLACE, R_.data(), n_ * n_, MPI_DOUBLE, MPI_SUM, mpi_comm_);
}

void cqr::gschol::savematrix(const char *filename, std::vector<double> &vec)
{

    std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (file.is_open())
    {   
        for(int i=0; i< vec.size(); ++i){
            file.write( reinterpret_cast<char*>( &vec[i] ), sizeof(double) );
        }

    }   
    else 
        std::cout << "File not opened!!!" << std::endl;
        file.close();
}

void cqr::gschol::vector_memset_zero(std::vector<double> &vec)
{
    memset(vec.data(), 0, vec.size() * sizeof vec[0]);
}
