__device__ __forceinline__
int search(
    const int * __restrict__ vec,
    const int search,
    const int length
) {
    int lower = 0;
    int upper = length;
    int middle;
    while (lower < upper) {
        middle = (lower + upper) >> 1;
        if (vec[middle] < search) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

__global__
void init(
    int * __restrict__ v1,
    int * __restrict__ v2,
    int numels
) {
    int i = getId();
    if (i >= numels) return;

    v1[i] = 2*i;
    v2[i] = 2*i + 1;
}

__global__
void merge(
    const int * __restrict__ v1,
    const int * __restrict__ v2,
    int * __restrict__ vmerge,
    int numels
) {
    // Iterativo
    /*
    for (int i = 0; i < numels; ++i) {
        i2 = search(v2, v1[i])
        vmerge[i+i2] = v1[i];
    }

    // tutti gli indici rimanenti [k] = v2[j]
    for (int i = 0; i < 2*numels; ++i) {
        if (vmerge[i] == -1) vmerge[i] = v2[i];
    }
    */
    int i = getId();
    if (i >= numels) return;

    int el1 = v1[i];
    // int el2 = v2[i];

    int index_el1_in_v2 = search(v2, el1, numels); // 2*i;
    // int index_el2_in_v1 = search(v1, el2, numels); // i1+1;

    vmerge[i+index_el1_in_v2] = el1;
    // vmerge[i+index_el2_in_v1] = el2;

    // sincronizza e aggiungi i mancanti
    // __syncthreads();
    // if (i == 0) {
    // 	for (int j = 0, j2 = 0; (j < 2*numels) && (j2 < numels); ++j) {
    // 		if (vmerge[j] == -1) {
    // 			vmerge[j] = v2[j2];
    // 			++j2;
    // 		}
    // 	}
    // }

    // DEBUG
    // vmerge[i] = i;
    // vmerge[i+numels] = i+numels;
}

// prende tutti quelli con -1 e setta v2
__global__
void filter(
    const int * __restrict__ input, // vmerge
    // const int * __restrict__ predicate,
    int * __restrict__ output,
    int numels
) {
    int i = getId();
    if (i >= numels) return;

    if (input[i] == -1) {
        output[i] = 1;
    } else {
        output[i] = 0;
    }

    // __syncthreads();

    // TODO: scan

}

__global__
void compact(
    const int * __restrict__ merge,  // 0  -1   2  -1   4  -1   6  -1
    const int * __restrict__ indexs, // 0   0   1   1   2   2   3   3
    const int * __restrict__ v2,     // 1   3   5   7
    int * __restrict__ output,
    int numels
) {
    int i = getId();
    if (i >= numels) return;

    if (merge[i] == -1) {
        const int idx = indexs[i];
        output[idx] = v2[idx];
    }
}

__global__
void finalize_merge(
    const int * __restrict__ v2,
    const int * __restrict__ compactedIndexs,
    int * __restrict__ merge,
    int numels
) {
    int i = getId();
    if (i >= numels) return;

    int idx = compactedIndexs[i];
    merge[idx] = v2[i];
}


extern __shared__ int shmem[];

__global__
void scan(const int4 * __restrict__ v1,
	int4 * __restrict__ vscan,
	int * __restrict__ code,
	int numels /*numero di elementi, multiplo di 4 */)
{
	const int window_size = blockDim.x;
    const int wi = threadIdx.x;
	const int numquarts = numels/4;
	const int quarts_per_block = div_up(numquarts, gridDim.x);
	/* prima quartina del blocco */
	const int first_quart = blockIdx.x*quarts_per_block;
	const int scan_end = min(first_quart + quarts_per_block, numquarts);
	/* coda della sliding window */
	int sliding_tail = 0;
	for (int i = first_quart + wi; i < scan_end; i += window_size) {
        int4 datum = v1[i];

        if (gridDim.x > 1) {
            shmem[threadIdx.x] = datum.w + datum.z + datum.y + datum.x;

            // 0 1 2 3
            datum.w = datum.z + datum.y + datum.x;
            // 0 1 2 3
            datum.z = datum.y + datum.x;
            // 0 1 1 3
            datum.y = datum.x;
            // 0 0 1 3
            datum.x = 0;
            // 0 0 1 3
        } else {
            // 1 2 3 4
            datum.y += datum.x;
            // 1 3 3 4
            datum.w += datum.z;
            // 1 3 3 7
            datum.z += datum.y;
            // 1 3 6 7
            datum.w += datum.y;
            // 1 3 6 10

            shmem[threadIdx.x] = datum.w;
        }

		__syncthreads();

		/* scansione delle code */
		for (int n = 1; n < blockDim.x ; n*=2) {
			if (threadIdx.x < blockDim.x/2) {
				int idx_r = (threadIdx.x/n)*(2*n) + (n-1);
				int idx_w = idx_r + 1 + (threadIdx.x & (n-1));
				shmem[idx_w] += shmem[idx_r];
			}
			__syncthreads();
		}

		/* correzione */
		int corr = sliding_tail;
		if (threadIdx.x > 0)
			corr += shmem[threadIdx.x-1];
		sliding_tail += shmem[blockDim.x-1];
		__syncthreads();
		datum.x += corr;
		datum.y += corr;
		datum.z += corr;
		datum.w += corr;
		vscan[i] = datum;
	}
	if (gridDim.x > 1 && threadIdx.x == 0)
		code[blockIdx.x] = sliding_tail;
}

__global__
void finalize_scan(int4 *vscan, int *code, int numels)
{
	const int window_size = blockDim.x;
	const int wi = threadIdx.x;
	const int numquarts = numels/4;
	const int quarts_per_block = div_up(numquarts, gridDim.x);
	/* prima quartina del blocco */
	const int first_quart = blockIdx.x*quarts_per_block;
	const int scan_end = min(first_quart + quarts_per_block, numquarts);
	/* correzione degli scan parziali */
	if (blockIdx.x == 0) return;
	int corr = code[blockIdx.x - 1];
	for (int i = first_quart + wi; i < scan_end; i += window_size) {
		int4 datum = vscan[i];
		datum.x += corr;
		datum.y += corr;
		datum.z += corr;
		datum.w += corr;
		vscan[i] = datum;
	}
}