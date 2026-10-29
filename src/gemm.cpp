#include <cstdio>

#include "kernels.hpp"
#include "utils.hpp"

int main(int argc, char **argv)
{
    int M, N, K;

    if (argc == 2)
        M = N = K = atoi(argv[1]);
    else
    {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    float *hA, *hB, *hC, *hRef;
    float *dA, *dB, *dC, *dRef;
    create_data(&hA, &hB, &hC, &hRef, &dA, &dB, &dC, &dRef, M, N, K);

    launch_gemm_ref(dA, dB, dRef, M, N, K);
    HIP_CHECK(hipMemcpy(hRef, dRef, M * N * sizeof(float), hipMemcpyDeviceToHost));

    auto run_kernel = [&](const char *name, auto launch_fn) {
        HIP_CHECK(hipMemset(dC, 0, M * N * sizeof(float)));
        float ms = launch_fn(dA, dB, dC, M, N, K);
        HIP_CHECK(hipMemcpy(hC, dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

        double err = max_rel_err(hRef, hC, M * N);
        bool ok = err < 1;
        double gflops = 2.0 * M * N * K / (ms * 1e6);

        printf("%-16s %s %8.4f ms %8.1f GFLOP/s (max rel err %.2g)\n", name,
               ok ? "PASS" : "FAIL", ms, gflops, err);
        return ok;
    };

    printf("Benchmarking GEMM kernels (M=%d N=%d K=%d)\n", M, N, K);
    bool all_ok = true;

    all_ok &= run_kernel("v00", launch_gemm_v00);
    all_ok &= run_kernel("v02", launch_gemm_kernel_v02<float>);
    all_ok &= run_kernel("v03", launch_gemm_kernel_v03<float>);
    all_ok &= run_kernel("v04", launch_gemm_kernel_v04<float>);
    all_ok &= run_kernel("v05", launch_gemm_kernel_v05<float>);
    all_ok &= run_kernel("v06", launch_gemm_kernel_v06<float>);
    all_ok &= run_kernel("v07", launch_gemm_kernel_v07<float>);

    printf("Summary: %s\n", all_ok ? "all kernels passed" : "some kernels failed");

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipFree(dRef));
    free(hA);
    free(hB);
    free(hC);
    free(hRef);
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
