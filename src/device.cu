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
}

// prende tutti quelli con -1 e setta v2
__global__
void filter(
    const int * __restrict__ input, // vmerge
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
    int * __restrict__ output,
    int numels
) {
    int i = getId();
    if (i >= numels) return;

    if (merge[i] == -1) {
        const int idx = indexs[i-1]; // -1 perché sto facendo scan inclusivo
        output[idx] = i;
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

__device__ __forceinline__
int4& operator+=(int4& v, int c)
{
	v.x += c; v.y += c;
	v.z += c; v.w += c;
	return v;
}

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
	const int4 nodata = make_int4(0, 0, 0, 0);
	for (int i = first_quart + wi; i < scan_end; i += 4*window_size) {
		int idx0 = i;
		int idx1 = i + window_size;
		int idx2 = i + 2*window_size;
		int idx3 = i + 3*window_size;
		int4 d0 = idx0 < scan_end ? v1[idx0] : nodata;
		int4 d1 = idx1 < scan_end ? v1[idx1] : nodata;
		int4 d2 = idx2 < scan_end ? v1[idx2] : nodata;
		int4 d3 = idx3 < scan_end ? v1[idx3] : nodata;
#define SCAN_V4(d) do { \
		d.y += d.x; \
		d.w += d.z; \
		d.z += d.y; \
		d.w += d.y; \
} while (0)

		SCAN_V4(d0);
		SCAN_V4(d1);
		SCAN_V4(d2);
		SCAN_V4(d3);

		shmem[threadIdx.x] = d0.w;
		shmem[threadIdx.x + blockDim.x] = d1.w;
		shmem[threadIdx.x + 2*blockDim.x] = d2.w;
		shmem[threadIdx.x + 3*blockDim.x] = d3.w;
		__syncthreads();

		/* scansione delle code */
		int idx_r = threadIdx.x*2;
#pragma unroll 8
		for (int n = 1; n < blockDim.x ; n*=2) {
			int idx_w = idx_r + 1 + (threadIdx.x & (n-1));
			shmem[idx_w] += shmem[idx_r];
			shmem[idx_w + 2*blockDim.x] += shmem[idx_r + 2*blockDim.x];
			int multiplo = !!(threadIdx.x & n);
			idx_r += n*(1 - multiplo) - n*multiplo;
			__syncthreads();
		}

		/* correzione */
		int shtail1 = shmem[blockDim.x - 1];
		int shtail2 = shmem[2*blockDim.x - 1];
		int shtail3 = shmem[3*blockDim.x - 1];
		int shtail4 = shmem[4*blockDim.x - 1];
		int corr0 = sliding_tail;
		int corr1 = sliding_tail + shtail1;
		int corr2 = sliding_tail + shtail1 + shtail2;
		int corr3 = (sliding_tail + shtail1) + (shtail2 + shtail3);
		if (threadIdx.x > 0) {
			corr0 += shmem[threadIdx.x                - 1];
			corr1 += shmem[threadIdx.x +   blockDim.x - 1];
			corr2 += shmem[threadIdx.x + 2*blockDim.x - 1];
			corr3 += shmem[threadIdx.x + 3*blockDim.x - 1];
		}
		sliding_tail += shtail1 + shtail2;
		sliding_tail += shtail3 + shtail4;
		__syncthreads();

		d0 += corr0;
		d1 += corr1;
		d2 += corr2;
		d3 += corr3;

		if (idx0 < scan_end) vscan[idx0] = d0;
		if (idx1 < scan_end) vscan[idx1] = d1;
		if (idx2 < scan_end) vscan[idx2] = d2;
		if (idx3 < scan_end) vscan[idx3] = d3;
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