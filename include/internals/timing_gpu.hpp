/*
 * File:   timing_gpu.hpp
 * Date:   July 7, 2023
 * Brief:  Definition and implementation of timing functions for GPU code.
 * 
 * This file is part of the CholeskyQR2-IM library.
 * 
 * Copyright (c) 2023-2024 Centre for Informatics and Computing,
 * Rudjer Boskovic Institute, Croatia. All rights reserved.
 * 
 * License: 3-clause BSD (BSD License 2.0)
 */


#include <cuda_runtime.h>

#pragma once

class TimingGpu
{
    private:
        std::map<std::string, float > durationmap_;
        std::map<std::string, std::vector<cudaEvent_t>> start_time_map_;
        std::map<std::string, std::vector<cudaEvent_t>> stop_time_map_;
        cudaEvent_t final_stop_;

        

    public:
        void start_timing(std::string time_name)
        {
            cudaEvent_t start;
            cudaEventCreate(&start);
            cudaEventRecord(start);
            start_time_map_[time_name].push_back(start);
        }

        void start_timing(std::string time_name, cudaStream_t  stream)
        {
            cudaEvent_t start;
            cudaEventCreate(&start);
            cudaEventRecord(start, stream);
            start_time_map_[time_name].push_back(start);
        }

        void stop_timing(std::string time_name)
        {
            cudaEvent_t stop;
            cudaEventCreate(&stop);
            cudaEventRecord(stop);
            stop_time_map_[time_name].push_back(stop);
            final_stop_ = stop;
        }

        void stop_timing(std::string time_name, cudaStream_t  stream)
        {
            cudaEvent_t stop;
            cudaEventCreate(&stop);
            cudaEventRecord(stop, stream);
            stop_time_map_[time_name].push_back(stop);
            final_stop_ = stop;
        }

        void print()
        {
                cudaEventSynchronize(final_stop_);
                float tmp_milisec = 0;

                for(auto &it: start_time_map_)
                {
                    size_t vecSize = start_time_map_[it.first].size();
                    for(size_t vecit = 0; vecit < vecSize; ++vecit)
                    {
                        cudaEventElapsedTime(&tmp_milisec, start_time_map_[it.first][vecit], stop_time_map_[it.first][vecit]);

                        if (auto search = durationmap_.find(it.first); search != durationmap_.end())
                        {
                            durationmap_[it.first] += tmp_milisec;
                        }
                        else
                            durationmap_[it.first]  = tmp_milisec;

                    }   
                }

                for(auto &it: durationmap_)
                {
                    std::cout << it.first << ": Value: " << it.second << std::endl;
                }
        }
};