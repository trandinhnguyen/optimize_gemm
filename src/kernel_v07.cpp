#include "kernels.hpp"
#include "load_data_to_shared_mem.hpp"
#include "utils.hpp"

using float4v = __attribute__((__vector_size__(4 * sizeof(float)))) float;

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int WARP_TILE_SIZE_X, int WARP_TILE_SIZE_Y,
          int NUM_WARPS_X, int NUM_WARPS_Y, int WMMA_TILE_SIZE_X,
          int WMMA_TILE_SIZE_Y, int WMMA_TILE_SIZE_K, int NUM_WMMA_TILES_X,
          int NUM_WMMA_TILES_Y, int NUM_WMMA_TILES_K, int NUM_THREADS,
          int BLOCK_SKEW_X = 0, int BLOCK_SKEW_Y = 0>
__global__ void gemm_v07(T const *A, T const *B, T *C, int M, int N, int K)
{
    __shared__ T
        A_s_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_SKEW_Y];
    __shared__ T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_SKEW_X];

    T A_vals[NUM_WMMA_TILES_Y] = {static_cast<T>(0)};
    T B_vals[NUM_WMMA_TILES_X] = {static_cast<T>(0)};
    float4v C_results[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X] = {0};

    int warp_linear_idx = threadIdx.x / 64;
    int warp_row_idx = warp_linear_idx / NUM_WARPS_X;
    int warp_col_idx = warp_linear_idx % NUM_WARPS_X;

    int shared_row = warp_row_idx * WARP_TILE_SIZE_Y;
    int shared_col = warp_col_idx * WARP_TILE_SIZE_X;

    int thread_idx_in_warp = threadIdx.x % 64;
    int thread_row_in_wmma = thread_idx_in_warp / 16;
    int thread_col_in_wmma = thread_idx_in_warp % 16;

    for (int block_tile_idx = 0; block_tile_idx < cdiv(K, BLOCK_TILE_SIZE_K);
         block_tile_idx++)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS, BLOCK_SKEW_X, BLOCK_SKEW_Y>(A, B, A_s_transposed, B_s, block_tile_idx, threadIdx.x,
                         M, N, K);
        __syncthreads();

#pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K;
             k_idx += WMMA_TILE_SIZE_K)
        {

#pragma unroll
            for (int wmma_tile_row_idx = 0;
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; wmma_tile_row_idx++)
            {
                A_vals[wmma_tile_row_idx] =
                    A_s_transposed[k_idx + thread_row_in_wmma]
                                  [shared_row +
                                   wmma_tile_row_idx * WMMA_TILE_SIZE_Y +
                                   thread_col_in_wmma];
            }

#pragma unroll
            for (int wmma_tile_col_idx = 0;
                 wmma_tile_col_idx < NUM_WMMA_TILES_X; wmma_tile_col_idx++)
            {
                B_vals[wmma_tile_col_idx] =
                    B_s[k_idx + thread_row_in_wmma]
                       [shared_col + wmma_tile_col_idx * WMMA_TILE_SIZE_X +
                        thread_col_in_wmma];
            }

#pragma unroll
            for (int wmma_tile_row_idx = 0;
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; wmma_tile_row_idx++)
            {
#pragma unroll
                for (int wmma_tile_col_idx = 0;
                     wmma_tile_col_idx < NUM_WMMA_TILES_X; wmma_tile_col_idx++)
                {
                    C_results[wmma_tile_row_idx][wmma_tile_col_idx] =
                        __builtin_amdgcn_mfma_f32_16x16x4f32(
                            A_vals[wmma_tile_row_idx],
                            B_vals[wmma_tile_col_idx],
                            C_results[wmma_tile_row_idx][wmma_tile_col_idx], 0,
                            0, 0);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int wmma_tile_row_idx = 0; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         wmma_tile_row_idx++)
    {
        int C_row = blockIdx.y * BLOCK_TILE_SIZE_Y + shared_row +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y +
                    thread_row_in_wmma * 4;

#pragma unroll
        for (int wmma_tile_col_idx = 0; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             wmma_tile_col_idx++)
        {
            int C_col = blockIdx.x * BLOCK_TILE_SIZE_X + shared_col +
                        wmma_tile_col_idx * WMMA_TILE_SIZE_X +
                        thread_col_in_wmma;

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                C[(C_row + i) * N + C_col] =
                    C_results[wmma_tile_row_idx][wmma_tile_col_idx][i];
            }
        }
    }
}

template <typename T>
float launch_gemm_kernel_v07(T *dA, T *dB, T *dC, int M, int N, int K)
{
    printf("Block tiling + warp tiling + matrix core + vectorized load + matrix tranpose\n");
    constexpr int BLOCK_TILE_SIZE_X = 128;
    constexpr int BLOCK_TILE_SIZE_Y = 128;
    constexpr int BLOCK_TILE_SIZE_K = 16;

    constexpr int WARP_TILE_SIZE_X = 64;
    constexpr int WARP_TILE_SIZE_Y = 64;

    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X; // 2
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y; // 2

    constexpr int WMMA_TILE_SIZE_X = 16;
    constexpr int WMMA_TILE_SIZE_Y = 16;
    constexpr int WMMA_TILE_SIZE_K = 4;

    constexpr int NUM_WMMA_TILES_X = WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X;
    constexpr int NUM_WMMA_TILES_Y = WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y;
    constexpr int NUM_WMMA_TILES_K = BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K;

    constexpr int NUM_THREADS = NUM_WARPS_X * NUM_WARPS_Y * 64; // = 256

    constexpr int BLOCK_SKEW_X = 4;
    constexpr int BLOCK_SKEW_Y = 4;

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(cdiv(N, BLOCK_TILE_SIZE_X), cdiv(M, BLOCK_TILE_SIZE_Y));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
    {
        gemm_v07<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                 WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, NUM_WARPS_X, NUM_WARPS_Y,
                 WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y, WMMA_TILE_SIZE_K,
                 NUM_WMMA_TILES_X, NUM_WMMA_TILES_Y, NUM_WMMA_TILES_K,
                 NUM_THREADS, BLOCK_SKEW_X, BLOCK_SKEW_Y>
            <<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}

template float launch_gemm_kernel_v07<float>(float *dA, float *dB, float *dC,
                                             int M, int N, int K);
