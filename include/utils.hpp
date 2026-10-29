#ifndef UTILS
#define UTILS

#include <hip/hip_runtime.h>

#define N_ITER 5

#define HIP_CHECK(cmd)                                                         \
    do                                                                         \
    {                                                                          \
        hipError_t _e = (cmd);                                                 \
        if (_e != hipSuccess)                                                  \
        {                                                                      \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),  \
                    __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__device__ __host__ inline int cdiv(int a, int b) { return (a + b - 1) / b; }

double max_rel_err(float *ref, float *got, int n);

void create_data(float **hA, float **hB, float **hC, float **hRef, float **dA,
                 float **dB, float **dC, float **dRef, int M, int N, int K);

#endif // UTILS