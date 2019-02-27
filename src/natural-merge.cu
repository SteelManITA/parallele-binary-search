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

    int *d_v1, *d_v2, *d_vmerge, *d_vscan;
    int *vmerge, *vscan;

    cudaError_t err;

    // memory alloc
    err = cudaMalloc(&d_v1, vSize);
    cudaCheck(err, "alloc v1");
    err = cudaMalloc(&d_v2, vSize);
    cudaCheck(err, "alloc v2");
    err = cudaMalloc(&d_vmerge, vmergeSize);
    cudaCheck(err, "alloc vmerge");
    err = cudaMalloc(&d_vscan, vmergeSize);
    cudaCheck(err, "alloc vscan");

    vmerge = new int[2 * numels];
    vscan = new int[2 * numels];

    if (!vmerge || !vscan) {
        error("impossibile allocare tutti i vettori");
    }

    err = cudaHostRegister(
        vmerge,
        vmergeSize,
        cudaHostRegisterPortable
    );
    cudaCheck(err, "pin vmerge");

    err = cudaHostRegister(
        vscan,
        vmergeSize,
        cudaHostRegisterPortable
    );
    cudaCheck(err, "pin vscan");

    int blockSize = 256;
    int numBlocks = (numels + blockSize - 1)/blockSize;
    int numBlocks2 = (numels*2 + blockSize - 1)/blockSize;

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
    err = cudaMemset(d_vscan, -1, vmergeSize);
    cudaCheck(err, "memset vscan");

    cudaRunEvent(
        "init",
        [&](){ init<<<numBlocks, blockSize>>>(d_v1, d_v2, numels); },
        2 * vSize
    );

    cudaRunEvent(
        "merge v1",
        [&](){ merge<<<numBlocks, blockSize>>>(d_v1, d_v2, d_vmerge, numels); },
        (2 * vSize) + (2 * vmergeSize) + (2 * vSize * (log(numels)/log(2))) // TODO: verificare
    );

    /* START SCAN */
    int blockSizeScan = 1024;
    int numBlocksScan = 8;
    int *d_code;
    cudaCheck(cudaMalloc(&d_code, numBlocksScan*sizeof(*d_code)),
        "allocazione code");

    cudaRunEvent(
        "scan",
        [&](){
            scan<<<numBlocksScan, blockSizeScan, 4*blockSizeScan*sizeof(int)>>>(
                (int4*)d_vmerge, // unico input
                (int4*)d_vscan,
                d_code, /* code */
                numels*2);
            if (numBlocksScan > 1) {
                scan<<<1, blockSizeScan, 4*blockSizeScan*sizeof(int)>>>(
                    (int4*)d_code,
                    (int4*)d_code,
                    NULL,
                    numBlocksScan);
                finalize_scan<<<numBlocksScan, blockSizeScan>>>(
                    (int4*)d_vscan,
                    d_code, /* code */
                    numels*2);
            }
        },
        0 // TODO: verificare
    );
    /* END SCAN */

    // int * d_vidx = d_v1;
    int * vidx = new int[numels];

    // uso d_v1 come appoggio (sporco l'input)
    cudaRunEvent(
        "finalize_merge v2",
        [&](){ finalize_merge<<<numBlocks2, blockSize>>>(d_vmerge, d_vscan, d_v2, 2*numels); },
        0 // TODO: verificare
    );

    cudaRunEvent(
        "cpy vmerge",
        [&](){ cudaMemcpy(vmerge, d_vmerge, vmergeSize, cudaMemcpyDeviceToHost); },
        vmergeSize
    );

    verify(vmerge, 2 * numels);

    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_vmerge);
    cudaFree(d_vscan);
    delete [] vmerge;
    delete [] vscan;

    return 0;
}

void verify(const int *vmerge, int numels)
{
    for (int i = 0; i < numels; ++i) {
        if (vmerge[i] != i) {
            fprintf(stderr, "vmerge: mismatch @ %d: %d != %d\n", i, vmerge[i], i);
            exit(2);
        }
    }
}
