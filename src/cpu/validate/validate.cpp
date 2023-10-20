#include <cmath>
#include <iostream>
#include <fstream>
#include <thread>

#include "validate.hpp"


double Validate::orthogonality()
{
    double norm = -14.0;
    double alpha = 1.0, beta = 0.0;

    std::vector<double> C(n_ * n_);
    dgemm("N", "T",
          &n_, &n_, &m_, 
          &alpha, Q_, &n_, 
          Q_, &n_,
          &beta, C.data(), &n_);

    MPI_Allreduce(MPI_IN_PLACE, C.data(), n_*n_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


    std::vector<double> I(n_, 1);   

    alpha = -1.0;
    std::int64_t incx = 1, incy = n_+1;
    daxpy(&n_, &alpha, I.data(), &incx, C.data(), &incy);

    std::int64_t n2 = n_ * n_;
    norm = dnrm2( &n2, C.data(), &incx);

    return (norm / std::sqrt(n_));
}


double Validate::residuals(std::vector<double> &A)
{
    double norm = -14.0;
    double normA = -14.0;

    double alpha = 1.0;
    double beta = 0.0;

    std::int64_t incx = 1; 

    dtrmm("L", "L", "N", "N",
          &n_, &m_, 
          &alpha, R_, &n_,
          Q_, &n_);


    normA = dnrm2( &size_, A.data(), &incx);
    normA *= normA;
    MPI_Allreduce(MPI_IN_PLACE, &normA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    normA = std::sqrt(normA);
        
    alpha = -1.0;
    daxpy(&size_, &alpha, A.data(), &incx, Q_, &incx);

    norm = dnrm2( &size_, Q_, &incx);
    norm *= norm;
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    norm = std::sqrt(norm);

    return (norm/normA);                      

}