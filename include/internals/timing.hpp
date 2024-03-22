/*
 * File:   timing.hpp
 * Date:   July 7, 2023
 * Brief:  Definition and implementation of timing functions for CPU-only code.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <iterator>
#include <map>
#include <string>
#include <iostream>
#include <chrono>
#ifdef GPU
    #include "timing_gpu.hpp"
#endif

#pragma once

class Timing
{
    using duration   = std::chrono::duration<double, std::milli>;
    using time_point = std::chrono::high_resolution_clock::time_point;

    private:
        std::map<std::string, duration > durationmap_;
        std::map<std::string, time_point > start_time_map_;

    public:
        void start_timing(std::string time_name)
        {
            start_time_map_[time_name] = std::chrono::high_resolution_clock::now();
        }

        void stop_timing(std::string time_name)
        {
            if (auto search = durationmap_.find(time_name); search != durationmap_.end())
                durationmap_[time_name] += std::chrono::high_resolution_clock::now() - start_time_map_[time_name];
            else
                durationmap_[time_name]  = std::chrono::high_resolution_clock::now() - start_time_map_[time_name];
        }

        void print()
        {
                std::map<std::string, duration>::iterator it = durationmap_.begin();
                while (it != durationmap_.end())
                {
                    std::cout << "Part: " << it->first << ", Value: " << it->second.count() << std::endl;
                    ++it;
                }
        }
};