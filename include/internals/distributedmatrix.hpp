/*
 * File:   distributedmatrix.hpp
 * Date:   July 7, 2023
 * Brief:  Definition of functions to distribute the input matrix among MPI processes.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <cstdint>
#include <vector>
#include <cmath>

#include <mpi.h>

#pragma once

class DistributedMatrix
{
    public:
        DistributedMatrix(std::int64_t m, std::int64_t n, MPI_Comm Comm);

        MPI_Datatype get_datatype();
        std::vector<int> get_displacements();
        std::vector<int> get_counts();
        
        std::int64_t get_rank_localm();
        int get_rank_displacement();
        int get_rank_count();
        


    private:
        void count_and_dispplacement();
        
        int world_size_;
        int world_rank_;
        std::int64_t m_, n_, localm_;
    
        std::vector<int> counts;
        std::vector<int> displacements;
        
        MPI_Comm mpi_comm_;
        MPI_Datatype row_datatype;
};