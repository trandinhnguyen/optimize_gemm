#include "kernels.hpp"
#include "load_data_to_shared_mem.hpp"
#include "utils.hpp"

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_Y, int NUM_THREADS>
__global__ void gemm_v03(T const *A, T const *B, T *C, int M, int N, int K)
{
    __shared__ T A_s[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    int shared_row = (threadIdx.x / BLOCK_TILE_SIZE_X) * THREAD_TILE_SIZE_Y;
    int shared_col = threadIdx.x % BLOCK_TILE_SIZE_X;

    T C_result[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

    for (int block_tile_idx = 0; block_tile_idx < cdiv(K, BLOCK_TILE_SIZE_K);
         block_tile_idx++)
    {
        load_data_to_shared_mem<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, B, A_s, B_s, block_tile_idx, threadIdx.x, M, N, K);
        __syncthreads();

#pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K; k_idx++)
        {
            T B_val = B_s[k_idx][shared_col];

#pragma unroll
            for (int thread_tile_idx = 0; thread_tile_idx < THREAD_TILE_SIZE_Y;
                 thread_tile_idx++)
            {
                C_result[thread_tile_idx] +=
                    A_s[shared_row + thread_tile_idx][k_idx] * B_val;
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int thread_tile_idx = 0; thread_tile_idx < THREAD_TILE_SIZE_Y;
         thread_tile_idx++)
    {
        int C_row =
            blockIdx.y * BLOCK_TILE_SIZE_Y + shared_row + thread_tile_idx;
        int C_col = blockIdx.x * BLOCK_TILE_SIZE_X + shared_col;

        if (C_row < M && C_col < N)
            C[C_row * N + C_col] = C_result[thread_tile_idx];
    }
}

template <typename T>
float launch_gemm_kernel_v03(T *dA, T *dB, T *dC, int M, int N, int K)
{
    printf("Block tiling + 1D Thread tiling\n");
    constexpr int BLOCK_TILE_SIZE_X = 64;
    constexpr int BLOCK_TILE_SIZE_Y = 64;
    constexpr int BLOCK_TILE_SIZE_K = 8;

    constexpr int THREAD_TILE_SIZE_Y = 16;
    constexpr int NUM_THREADS =
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y;

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(cdiv(N, BLOCK_TILE_SIZE_X), cdiv(M, BLOCK_TILE_SIZE_Y));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
    {
        gemm_v03<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                 THREAD_TILE_SIZE_Y, NUM_THREADS>
            <<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}

template float launch_gemm_kernel_v03<float>(float *dA, float *dB, float *dC,
                                             int M, int N, int K);
