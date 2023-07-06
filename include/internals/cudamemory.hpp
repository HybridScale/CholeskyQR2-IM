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
};