#ifndef KERNELS
#define KERNELS

#include <hip/hip_runtime.h>

void launch_gemm_ref(float *dA, float *dB, float *dRef, int M, int N, int K);
float launch_gemm_v00(float *dA, float *dB, float *dC, int M, int N, int K);

template <typename T>
float launch_gemm_kernel_v02(T *dA, T *dB, T *dC, int M, int N, int K);
template <typename T>
float launch_gemm_kernel_v03(T *dA, T *dB, T *dC, int M, int N, int K);
template <typename T>
float launch_gemm_kernel_v04(T *dA, T *dB, T *dC, int M, int N, int K);

template <typename T>
float launch_gemm_kernel_v05(T *dA, T *dB, T *dC, int M, int N, int K);

template <typename T>
float launch_gemm_kernel_v06(T *dA, T *dB, T *dC, int M, int N, int K);

template <typename T>
float launch_gemm_kernel_v07(T *dA, T *dB, T *dC, int M, int N, int K);
#endif