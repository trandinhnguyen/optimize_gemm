#include "kernels.hpp"
#include "load_data_to_shared_mem.hpp"
#include "utils.hpp"

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int WARP_TILE_SIZE_X, int WARP_TILE_SIZE_Y,
          int NUM_WARPS_X, int NUM_WARPS_Y, int THREAD_TILE_SIZE_X,
          int THREAD_TILE_SIZE_Y, int NUM_THREADS_PER_WARP_X,
          int NUM_THREADS_PER_WARP_Y, int NUM_THREADS_X, int NUM_THREADS_Y,
          int NUM_THREADS>
__global__ void gemm_v06(T const *A, T const *B, T *C, int M, int N, int K)
{

    __shared__ T A_s_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    T C_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {static_cast<T>(0)};
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    constexpr size_t NUM_VECTOR_UNITS = sizeof(int4) / sizeof(T);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X =
        THREAD_TILE_SIZE_X / NUM_VECTOR_UNITS;
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_Y =
        THREAD_TILE_SIZE_Y / NUM_VECTOR_UNITS;

    int warp_linear_idx = threadIdx.x / 64;
    int warp_row_idx = warp_linear_idx / NUM_WARPS_X;
    int warp_col_idx = warp_linear_idx % NUM_WARPS_X;

    int thread_idx_in_warp = threadIdx.x % 64;
    int thread_row_idx_in_warp = thread_idx_in_warp / NUM_THREADS_PER_WARP_X;
    int thread_col_idx_in_warp = thread_idx_in_warp % NUM_THREADS_PER_WARP_X;

    int shared_row = warp_row_idx * WARP_TILE_SIZE_Y +
                     thread_row_idx_in_warp * THREAD_TILE_SIZE_Y;
    int shared_col = warp_col_idx * WARP_TILE_SIZE_X +
                     thread_col_idx_in_warp * THREAD_TILE_SIZE_X;

    for (int block_tile_idx = 0; block_tile_idx < cdiv(K, BLOCK_TILE_SIZE_K);
         block_tile_idx++)
    {
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, B, A_s_transposed, B_s, block_tile_idx, threadIdx.x,
                         M, N, K);
        __syncthreads();

#pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_TILE_SIZE_K; k_idx++)
        {
            // A_s to A reg
#pragma unroll
            for (int thread_tile_row_vector_idx = 0;
                 thread_tile_row_vector_idx < VECTORIZED_THREAD_TILE_SIZE_Y;
                 thread_tile_row_vector_idx++)
            {
                *reinterpret_cast<int4 *>(
                    &A_vals[thread_tile_row_vector_idx * NUM_VECTOR_UNITS]) =
                    *reinterpret_cast<int4 const *>(
                        &A_s_transposed[k_idx][shared_row +
                                               thread_tile_row_vector_idx *
                                                   NUM_VECTOR_UNITS]);
            }

            // Read from B_s to B register is vectorized
#pragma unroll
            for (int thread_tile_col_vector_idx = 0;
                 thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                 thread_tile_col_vector_idx++)
            {
                *reinterpret_cast<int4 *>(
                    &B_vals[thread_tile_col_vector_idx * NUM_VECTOR_UNITS]) =
                    *reinterpret_cast<int4 const *>(
                        &B_s[k_idx][shared_col + thread_tile_col_vector_idx *
                                                     NUM_VECTOR_UNITS]);
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

// Vectorized writing the results to DRAM.
#pragma unroll
    for (int thread_tile_row = 0; thread_tile_row < THREAD_TILE_SIZE_Y;
         thread_tile_row++)
    {
#pragma unroll
        for (int thread_tile_col_vector_idx = 0;
             thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
             thread_tile_col_vector_idx++)
        {
            int C_row =
                blockIdx.y * BLOCK_TILE_SIZE_Y + shared_row + thread_tile_row;

            int C_col = blockIdx.x * BLOCK_TILE_SIZE_X + shared_col +
                        thread_tile_col_vector_idx * NUM_VECTOR_UNITS;

            // Assume the data alignment of C is a multiple of 16 bytes.
            *reinterpret_cast<int4 *>(&C[C_row * N + C_col]) =
                *reinterpret_cast<int4 *>(
                    &C_results[thread_tile_row]
                              [thread_tile_col_vector_idx * NUM_VECTOR_UNITS]);
        }
    }
}

template <typename T>
float launch_gemm_kernel_v06(T *dA, T *dB, T *dC, int M, int N, int K)
{
    printf("Block tiling + warp tiling + thread tiling + vectorized load + matrix transposed\n");
    constexpr int BLOCK_TILE_SIZE_X = 128;

    constexpr int BLOCK_TILE_SIZE_Y = 128;
    constexpr int BLOCK_TILE_SIZE_K = 16;

    constexpr int WARP_TILE_SIZE_X = 64;
    constexpr int WARP_TILE_SIZE_Y = 64;

    constexpr int NUM_WARPS_X = BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X; // 2
    constexpr int NUM_WARPS_Y = BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y; // 2

    constexpr int THREAD_TILE_SIZE_X = 8;
    constexpr int THREAD_TILE_SIZE_Y = 8;

    constexpr int NUM_THREADS_PER_WARP_X = 8;
    constexpr int NUM_THREADS_PER_WARP_Y = 8;

    constexpr int NUM_THREADS_X = NUM_WARPS_X * NUM_THREADS_PER_WARP_X; // = 16
    constexpr int NUM_THREADS_Y = NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y; // = 16

    constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y; // = 256

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(cdiv(N, BLOCK_TILE_SIZE_X), cdiv(M, BLOCK_TILE_SIZE_Y));

    hipEvent_t start, stop;
    float total_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < N_ITER; ++i)
    {
        gemm_v06<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
                 WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y, NUM_WARPS_X, NUM_WARPS_Y,
                 THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y, NUM_THREADS_PER_WARP_X,
                 NUM_THREADS_PER_WARP_Y, NUM_THREADS_X, NUM_THREADS_Y,
                 NUM_THREADS><<<dimGrid, dimBlock>>>(dA, dB, dC, M, N, K);
    }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / N_ITER;
}

template float launch_gemm_kernel_v06<float>(float *dA, float *dB, float *dC,
                                             int M, int N, int K);
