#include "kernels.hpp"
#include "load_data_to_shared_mem.hpp"
#include "utils.hpp"

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(T *dA, T *dB, T *dC, int M, int N, int K)
{
    constexpr int NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;

    const int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;

    const int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ T A_s[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    T sum = static_cast<T>(0);

    for (int tile_idx = 0; tile_idx < cdiv(K, BLOCK_TILE_SIZE_K); ++tile_idx)
    {
        load_data_to_shared_mem<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K, NUM_THREADS>(
            dA, dB, A_s, B_s, tile_idx, thread_linear_idx, M, N, K);
        __syncthreads();

#pragma unroll
        for (int k_i = 0; k_i < BLOCK_TILE_SIZE_K; k_i++)
            sum += A_s[threadIdx.y][k_i] * B_s[k_i][threadIdx.x];
        __syncthreads();
    }

    if (C_row < M && C_col < N)
        dC[C_row * N + C_col] = sum;
}

template <typename T>
float launch_gemm_kernel_v02(T *dA, T *dB, T *dC, int M, int N, int K)
{
    printf("Block tiling\n");
    constexpr int BLOCK_TILE_SIZE_X = 32;
    constexpr int BLOCK_TILE_SIZE_Y = 32;
    constexpr int BLOCK_TILE_SIZE_K = 32;
    constexpr int NUM_THREADS = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;

    dim3 dimBlock(BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y);
    dim3 dimGrid(cdiv(N, dimBlock.x), cdiv(M, dimBlock.y));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
    {
        gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
            <<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}

template float launch_gemm_kernel_v02<float>(float *dA, float *dB, float *dC,
                                             int M, int N, int K);
