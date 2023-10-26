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