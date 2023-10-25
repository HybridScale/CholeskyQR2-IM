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

    class gschol{
    public:
        gschol(std::int64_t m, std::int64_t n, std::int64_t panel_size);
        gschol(std::int64_t m, std::int64_t n, std::size_t panel_num);
        ~gschol();

        void InputMatrix(cudamemory<double> &A);
        void InputMatrix(double *A);
        void InputMatrix(std::string filename);

        void Start();
        void Validate_output();

     private:
#ifdef GPU
        void gschol2(cudamemory<double> &A, cudamemory<double> &R);
        //void gschol(cudamemory<double> &A, cudamemory<double> &R);
        //void save_R(double* R, double* tmp);
        void save_R(double* R, std::size_t ldr, double* tmp, std::size_t ldtmp, int m, int n);
        void update_R();
        void multiply_R(double* R, double* tmp);
        void reothrogonalize_panel(cudamemory<double> &A, int panel_number);
        void update_rest_Matrix(cudamemory<double> &A, int panel_number);

#else
        void gschol2(std::vector<double> &A, std::vector<double> &R);
        //void gschol(std::vector<double> &A, std::vector<double> &R);
#endif
        void MPI_Warmup();
        void first_panel_orth();
        void gramMatrix(double *A, double *R, double *tmp);
        void cholesky(double *B);
        void calculateQ(double *A, double *R);
        float get_time();

        std::int64_t n_, m_, localm_, block_size_;
        std::int64_t input_panel_size_, panel_size_;
        std::int64_t size = 1;
        std::string filename_;

        std::vector<double> A_;
        std::vector<double> Alocal_;
        std::vector<double> R_;

#ifdef GPU
        cudamemory<double> cudaAlocal_;
        cudamemory<double> cudaR_, cudaR1_, cudaR2_;
        cudamemory<double> cudaI_;
        cudamemory<double> cudatmp_, cudatmp2_, cudaWtmp_;
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

#ifdef NCCL
            ncclUniqueId NCCLid_;
            ncclComm_t nccl_comm_;
#endif
#endif
    };

}
