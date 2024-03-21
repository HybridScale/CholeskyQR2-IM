/*
 * File:   validate.hpp
 * Date:   July 7, 2023
 * Brief:  Definition of validation functions for CPU-only. Computes the residual and orthogonality of the obtained factors Q and R.
 * 
 * This file is part of the CholeskyQR2++ library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <cstdint>
#include <vector>

#include "mpi.h"

#define MKL_INT std::int64_t
#include "mkl.h"

#pragma once

class Validate
{
    private:
        std::int64_t m_, n_, size_;
        double *Q_, *R_;
        const char *filename_;
        
    public:
        Validate(std::int64_t m, std::int64_t n, double *Q, double *R, const char *file):
         m_(m), n_(n), Q_(Q), R_(R), filename_(file)
         {
            size_ = m_ * n_;
         };

        double orthogonality();
        double residuals(std::vector<double> &A);

};