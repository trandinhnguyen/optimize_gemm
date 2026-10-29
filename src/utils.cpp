#include "utils.hpp"
#include <random>

double max_rel_err(float *ref, float *got, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; i++)
        e = std::fmax(e,
                      std::fabs(got[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
    return e;
}

void create_data(float **hA, float **hB, float **hC, float **hRef, float **dA,
                 float **dB, float **dC, float **dRef, int M, int N, int K)
{
    int sizeA = M * K;
    int sizeB = K * N;
    int sizeC = M * N;

    *hA = (float *)malloc(sizeA * sizeof(float));
    *hB = (float *)malloc(sizeB * sizeof(float));
    *hC = (float *)malloc(sizeC * sizeof(float));
    *hRef = (float *)malloc(sizeC * sizeof(float));
    HIP_CHECK(hipMalloc(dA, sizeA * sizeof(float)));
    HIP_CHECK(hipMalloc(dB, sizeB * sizeof(float)));
    HIP_CHECK(hipMalloc(dC, sizeC * sizeof(float)));
    HIP_CHECK(hipMalloc(dRef, sizeC * sizeof(float)));

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (size_t i = 0; i < sizeA; ++i)
        (*hA)[i] = dist(gen);
    for (size_t i = 0; i < sizeB; ++i)
        (*hB)[i] = dist(gen);

    HIP_CHECK(
        hipMemcpy(*dA, *hA, sizeA * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(*dB, *hB, sizeB * sizeof(float), hipMemcpyHostToDevice));
}
