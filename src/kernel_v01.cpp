#include "kernels.hpp"
#include "utils.hpp"

#include <hip/hip_runtime.h>

#define REF_BLOCK_SIZE 32


__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    float s = 0;
    for (int k = 0; k < K; k++)
    {
        s += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = s;
}

void launch_gemm_ref(float *dA, float *dB, float *dRef, int M, int N, int K)
{
    dim3 dimBlock(REF_BLOCK_SIZE, REF_BLOCK_SIZE);
    dim3 dimGrid(cdiv(N, REF_BLOCK_SIZE), cdiv(M, REF_BLOCK_SIZE));
    matmul_kernel<<<dimGrid, dimBlock>>>(dA, dB, dRef, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

float launch_gemm_v00(float *dA, float *dB, float *dC, int M, int N, int K)
{
    printf("Naive Implementation\n");
    dim3 dimBlock(REF_BLOCK_SIZE, REF_BLOCK_SIZE);
    dim3 dimGrid(cdiv(N, REF_BLOCK_SIZE), cdiv(M, REF_BLOCK_SIZE));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
        matmul_kernel<<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}