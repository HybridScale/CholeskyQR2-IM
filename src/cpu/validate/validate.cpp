#include <cmath>
#include <iostream>
#include <fstream>
#include <thread>
#include "omp.h"

#include "validate.hpp"


double Validate::orthogonality()
{
    unsigned int processor_count = std::thread::hardware_concurrency();
    //omp_set_num_threads(processor_count);
    double norm = -14.0;
    double alpha = 1.0, beta = 0.0;

    std::vector<double> C(n_ * n_);
    dgemm("N", "T",
          &n_, &n_, &m_, 
          &alpha, Q_, &n_, 
          Q_, &n_,
          &beta, C.data(), &n_);


    std::vector<double> I(n_, 1);   

    alpha = -1.0;
    std::int64_t incx = 1, incy = n_+1;
    daxpy(&n_, &alpha, I.data(), &incx, C.data(), &incy);

    std::int64_t n2 = n_ * n_;
    norm = dnrm2( &n2, C.data(), &incx);

    return (norm / std::sqrt(n_));
}


double Validate::residuals()
{
    unsigned int processor_count = std::thread::hardware_concurrency();
    //omp_set_num_threads(processor_count);

    double norm = -14.0;
    double normA = -14.0;

    double alpha = 1.0;
    double beta = 0.0;

    std::int64_t incx = 1; 

    dtrmm("L", "L", "N", "N",
          &n_, &m_, 
          &alpha, R_, &n_,
          Q_, &n_);

    {
        std::vector<double> A(size_);
        InputMatrix(A);

        normA = dnrm2( &size_, A.data(), &incx);
        
        alpha = -1.0;
        daxpy(&size_, &alpha, A.data(), &incx, Q_, &incx);


    }  
    
    norm = dnrm2( &size_, Q_, &incx);

    return (norm/normA);                      

}


void Validate::InputMatrix(std::vector<double> &A)
{
    std::ifstream file(filename_, std::ios::in | std::ios::binary);
    
    if (file.is_open())
    {   
        for(int i=0; i< size_; ++i)
        {
            file.read( reinterpret_cast<char*>( &A[i] ), sizeof(double) );
        }
    }
    else
        std::cout << "File not opened while validate!!!" << std::endl;
}