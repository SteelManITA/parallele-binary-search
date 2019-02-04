extern __shared__ int shmem[];

__global__
void populateSample(
    int * array,
    int sample_length,
    int array_length
) {
    int i = getId();
    if (i >= sample_length) return;

    shmem[i] = array[i * (array_length / sample_length)];
    // if (i < 10 || i > sample_length - 10) {
    //     printf("\n%d * (%d / %d) = %d\t=>\t%d",
    //         i,
    //         array_length,
    //         sample_length,
    //         i * (array_length / sample_length),
    //         shmem[i]
    //     );
    //     // printf("\nsample[%d] = %d", i, shmem[i]);
    // }
}

__device__
int binary_search_guess(
    // const int * array,
    int number_of_elements,
    int key
) {
    // int lower = 0;
    // int upper = length;
    // int middle;
    // while (lower < upper) {
    //     middle = (lower + upper) >> 1;
    //     if (vec[middle] < search) {
    //         lower = middle + 1;
    //     } else {
    //         upper = middle;
    //     }
    // }
    // return lower;
    int lower = 0;
    int upper = number_of_elements;
    int middle;

    if (key == 0)
        printf("%d %d %d", key, shmem[lower], shmem[upper]);

    while (lower < upper) {
        middle = (lower + upper) >> 1;
        // inizio dei chunk


        // centro dei chunk
        if (key < shmem[middle]) {
            lower = middle + 1;
        } else {
            upper = middle;
        }


        // fine dei chunk


    }
    return lower;
}

__device__
int binary_search_precise(
    const int * array,
    int key,
    int low,
    int high
) {
	int mid;
	while(low <= high)
	{
		mid = (low + high)/2;
		if(array[mid] < key)
		{
			low = mid + 1;
		}
		else if(array[mid] == key)
		{
			return mid;
		}
		else if(array[mid] > key)
		{
			high = mid-1;
		}
	}
	return low;
}

__global__
void search2(
    const int * array,
    // int * sample,
    int * output,
    const int * query,
    int sample_length,
    int array_length
) {
    int i = getId();
    if (i >= array_length) return;

    if (i != 131071) return;
	// if(query[i] < *array || query[i] >= shmem[sample_length-1] + (array_length / sample_length)) {
    //     output[i] = -1;
    // }

    // int guess = binary_search_guess(sample_length,query[i]);
    int number_of_elements = sample_length;
    int key = query[i];


    int lower = 0;
    int upper = number_of_elements -1;
    int middle;

    printf("\nCerco %d tra shmem[%d] (%d) e shmem[%d] (%d)", key, lower, shmem[lower], upper, shmem[upper]);

    // if (key >= shmem[upper]) {
    //     printf("VALORE TROVATO: %d", upper);
    // }

    while (upper - lower >= 2) {
        middle = (lower + upper) >> 1;
        // inizio dei chunk


        // centro dei chunk
        if (key < shmem[middle]) {
            upper = middle;
        } else {
            lower = middle; // +1 solo al primo chunk
        }
    }

    // return lower;



    // printf("", lower);
    printf("\n%d < %d < %d => %d < %d < %d", lower, middle, upper, shmem[lower], key, shmem[upper]);
}


__device__ int get_index_to_check(int thread, int num_threads, int set_size, int offset) {

	// Integer division trick to round up
	return (((set_size + num_threads) / num_threads) * thread) + offset;
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
    int numels,
    int * vidx
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

    int i2 = vidx[i];

    int el = v1[i];
    vmerge[i+i2] = el;

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