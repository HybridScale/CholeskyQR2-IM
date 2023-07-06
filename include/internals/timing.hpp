#include <iterator>
#include <map>
#include <string>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

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