/*
 * File:   cudamemory.hpp
 * Date:   July 7, 2023
 * Brief:  Definition of GPU device memory operations.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <vector>
#include <cstdint> 

#pragma once

template <typename T>
class cudamemory
{
    private:
        T *cudaptr;
        std::int64_t m_;

    public:
        cudamemory();

        cudamemory(std::int64_t m);

        cudamemory(std::vector<T> &vec);

        ~cudamemory();

        T*  data();

        void resize(std::int64_t m);

        void memset();

        void memset(T value);

        void copytohost(std::vector<T> &host);

        void copytodevice(std::vector<T> &host);

        void savematrix(const char* filename);

        void release();
};