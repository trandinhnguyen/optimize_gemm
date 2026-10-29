#include "kernels.hpp"
#include "load_data_to_shared_mem.hpp"
#include "utils.hpp"

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int THREAD_TILE_SIZE_X, int THREAD_TILE_SIZE_Y,
          int NUM_THREADS>
__global__ void gemm_v04(T const *A, T const *B, T *C, int M, int N, int K)
{
    __shared__ T A_s[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    int shared_row = (threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X)) *
                     THREAD_TILE_SIZE_Y;
    int shared_col = (threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X)) *
                     THREAD_TILE_SIZE_X;

    T C_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

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
            // A_s to A reg
#pragma unroll
            for (int thread_tile_row = 0; thread_tile_row < THREAD_TILE_SIZE_Y;
                 thread_tile_row++)
            {
                A_vals[thread_tile_row] =
                    A_s[shared_row + thread_tile_row][k_idx];
            }

            // B_s to B regs
#pragma unroll
            for (int thread_tile_col = 0; thread_tile_col < THREAD_TILE_SIZE_X;
                 thread_tile_col++)
            {
                B_vals[thread_tile_col] =
                    B_s[k_idx][shared_col + thread_tile_col];
            }

            // Compute
#pragma unroll
            for (int thread_tile_row = 0; thread_tile_row < THREAD_TILE_SIZE_Y;
                 thread_tile_row++)
            {
#pragma unroll
                for (int thread_tile_col = 0;
                     thread_tile_col < THREAD_TILE_SIZE_X; thread_tile_col++)
                {
                    C_results[thread_tile_row][thread_tile_col] +=
                        A_vals[thread_tile_row] * B_vals[thread_tile_col];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int thread_tile_row = 0; thread_tile_row < THREAD_TILE_SIZE_Y;
         thread_tile_row++)
    {
#pragma unroll
        for (int thread_tile_col = 0; thread_tile_col < THREAD_TILE_SIZE_X;
             thread_tile_col++)
        {

            int C_row =
                blockIdx.y * BLOCK_TILE_SIZE_Y + shared_row + thread_tile_row;
            int C_col =
                blockIdx.x * BLOCK_TILE_SIZE_X + shared_col + thread_tile_col;

            if (C_row < M && C_col < N)
            {
                C[C_row * N + C_col] =
                    C_results[thread_tile_row][thread_tile_col];
            }
        }
    }
}

template <typename T>
float launch_gemm_kernel_v04(T *dA, T *dB, T *dC, int M, int N, int K)
{
    printf("Block tiling + 2D Thread tiling\n");
    constexpr int BLOCK_TILE_SIZE_X = 64;
    constexpr int BLOCK_TILE_SIZE_Y = 64;
    constexpr int BLOCK_TILE_SIZE_K = 16;

    constexpr int THREAD_TILE_SIZE_X = 4;
    constexpr int THREAD_TILE_SIZE_Y = 4;
    constexpr int NUM_THREADS = (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                                (BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y);

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(cdiv(N, BLOCK_TILE_SIZE_X), cdiv(M, BLOCK_TILE_SIZE_Y));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
    {
        gemm_v04<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                 THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y, NUM_THREADS>
            <<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}

template float launch_gemm_kernel_v04<float>(float *dA, float *dB, float *dC,
                                             int M, int N, int K);
