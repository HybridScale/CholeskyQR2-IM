#include <cuda_runtime.h>
#include "nvToolsExt.h"

#pragma once

#define CUDA_CHECK(call) {                               \
  cudaError_t cudaerr = call;                            \
  if( cudaerr != cudaSuccess ) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",            \
        __FILE__,__LINE__,cudaGetErrorString(cudaerr));  \
    exit(1); }                                           \
}                                     


#define CUBLAS_CHECK(call)                                          \
  if((call) != CUBLAS_STATUS_SUCCESS) {                             \
    printf("%s: %i CUBLAS error: %d\n", __FILE__, __LINE__, call);  \
  exit(1); }


#define CUSOLVER_CHECK(call)                                             \
    if((call) != CUSOLVER_STATUS_SUCCESS) {                              \
        printf("%s: %i CUSOLVER error: %d\n", __FILE__, __LINE__, call); \
        exit(1); }


#ifdef NCCL
    #include "nccl.h"
  #define NCCLCHECK(cmd){                             \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
      printf("Failed, NCCL error %s:%d '%s'\n",       \
          __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(1); }  \
    }
#endif

class NvtxTracer {
public:
    NvtxTracer(const char* name) {
        nvtxRangePushA(name);
    }
    ~NvtxTracer() {
        nvtxRangePop();
    }
};
