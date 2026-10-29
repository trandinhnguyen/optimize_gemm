#ifndef LOAD_DATA_TO_SHARED_MEM
#define LOAD_DATA_TO_SHARED_MEM

#include "utils.hpp"
#include <hip/hip_runtime.h>

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int NUM_THREADS>
__device__ void load_data_to_shared_mem(
    T const *A, T const *B, T A_s[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
    T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X], int tile_idx,
    int thread_linear_idx, int M, int N, int K)
{
#pragma unroll
    for (int load_idx = 0;
         load_idx < cdiv(BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K, NUM_THREADS);
         load_idx++)
    {
        int A_s_row =
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K;
        int A_s_col =
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K;

        int A_row = blockIdx.y * BLOCK_TILE_SIZE_Y + A_s_row;
        int A_col = tile_idx * BLOCK_TILE_SIZE_K + A_s_col;

        T val = static_cast<T>(0);
        if (A_row < M && A_col < K)
            val = A[A_row * K + A_col];

        A_s[A_s_row][A_s_col] = val;
    }

#pragma unroll
    for (int load_idx = 0;
         load_idx < cdiv(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X, NUM_THREADS);
         load_idx++)
    {
        int B_s_row =
            (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_X;
        int B_s_col =
            (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_X;

        int B_row = tile_idx * BLOCK_TILE_SIZE_K + B_s_row;
        int B_col = blockIdx.x * BLOCK_TILE_SIZE_X + B_s_col;

        T val = static_cast<T>(0);
        if (B_row < K && B_col < N)
            val = B[B_row * N + B_col];

        B_s[B_s_row][B_s_col] = val;
    }
}

template <typename T, int BLOCK_TILE_SIZE_X, int BLOCK_TILE_SIZE_Y,
          int BLOCK_TILE_SIZE_K, int NUM_THREADS, int BLOCK_SKEW_X = 0,
          int BLOCK_SKEW_Y = 0, typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(
    T const *A, T const *B,
    T A_s_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_SKEW_Y],
    T B_s[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_SKEW_X], int tile_idx,
    int thread_linear_idx, int M, int N, int K)
{
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};

    //   Load data from A on DRAM so shared mem
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         cdiv(BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K, NUM_THREADS);
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};

        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        int4 A_row_vector_vals{0, 0, 0, 0};
        // Assume A is aligned well
        A_row_vector_vals =
            *reinterpret_cast<int4 const *>(&A[A_row_idx * K + A_col_idx]);

#pragma unroll
        for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
        {
            A_s_transposed[A_thread_block_tile_col_idx +
                           i][A_thread_block_tile_row_idx] =
                reinterpret_cast<T const *>(&A_row_vector_vals)[i];
        }
    }

    // Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         cdiv(BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X, NUM_THREADS);
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
        size_t const B_row_idx{tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        int4 B_row_vector_vals{0, 0, 0, 0};
        B_row_vector_vals =
            *reinterpret_cast<int4 const *>(&B[B_row_idx * N + B_col_idx]);
        *reinterpret_cast<int4 *>(
            &B_s[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx]) =
            B_row_vector_vals;
    }
}

#endif // LOAD_DATA_TO_SHARED_MEM