#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>

#include <cuda_runtime_api.h>

#include "utils.cu"
#include "device.cu"

// INT_MAX byte allocabili dinamicamente (2GB)
// MAX_INPUT = INT_MAX / VEC_WEIGHT
#define VEC_COUNT 2
#define MAX_INPUT INT_MAX / 10 // TODO: verificare

void verify(const int *vmerge, int numels);

int main(int argc, char *argv[])
{
    int numels;

    if (argc != 2) {
        // error("sintassi: vecmerge numels");
        numels = MAX_INPUT;
    } else {
        numels = atoi(argv[1]);
        if (numels > MAX_INPUT) {
            numels = MAX_INPUT;
        }
        if (numels <= 0) {
            error("numels deve essere positivo");
        }
    }
    printf("numels set to %d\n", numels);

    const size_t vSize = numels*sizeof(int);
    const size_t vmergeSize = 2 * vSize;

    int *d_v1, *d_v2, *d_vmerge, *vmerge;

    cudaError_t err;

    // memory alloc
    err = cudaMalloc(&d_v1, vSize);
    cudaCheck(err, "alloc v1");
    err = cudaMalloc(&d_v2, vSize);
    cudaCheck(err, "alloc v2");
    err = cudaMalloc(&d_vmerge, vmergeSize);
    cudaCheck(err, "alloc vmerge");

    vmerge = new int[2 * numels];

    if (!vmerge) {
        error("impossibile allocare tutti i vettori");
    }

    err = cudaHostRegister(
        vmerge,
        vmergeSize,
        cudaHostRegisterPortable
    );
    cudaCheck(err, "pin vmerge");

    int blockSize = 256;
    int numBlocks = (numels + blockSize - 1)/blockSize;

    cudaDeviceProp devProps;
    err = cudaGetDeviceProperties(&devProps, 0);
    cudaCheck(err, "get device props");

    printf("grid: %d/%d\n", numBlocks, devProps.maxGridSize[0]);

    err = cudaMemset(d_v1, -1, vSize);
    cudaCheck(err, "memset v1");
    err = cudaMemset(d_v2, -1, vSize);
    cudaCheck(err, "memset v2");
    err = cudaMemset(d_vmerge, -1, vmergeSize);
    cudaCheck(err, "memset vmerge");

    cudaRunEvent(
        "init",
        [&](){ init<<<numBlocks, blockSize>>>(d_v1, d_v2, numels); },
        2 * vSize
    );

    cudaRunEvent(
        "merge",
        [&](){ merge<<<numBlocks, blockSize>>>((int4*)d_v1, (int4*)d_v2, d_vmerge, numels); },
        (2 * vSize / 4) + (8 * vmergeSize) + (2 * vSize * (log(numels/4)/log(2))) // TODO: verificare
    );

    cudaRunEvent(
        "cpy",
        [&](){ cudaMemcpy(vmerge, d_vmerge, vmergeSize, cudaMemcpyDeviceToHost); },
        vmergeSize
    );

    verify(vmerge, 2 * numels);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_vmerge);
    delete [] vmerge;

    return 0;
}

void verify(const int *vmerge, int numels)
{
    for (int i = 0; i < numels; ++i) {
        if (vmerge[i] != i) {
            fprintf(stderr, "mismatch @ %d: %d != %d\n", i, vmerge[i], i);
            exit(2);
        }
    }
}