__device__ __forceinline__
int search(
    const int * __restrict__ vec,
    int lower,
    int upper,
    const int search
) {
    int middle;
    while (lower < upper) {
        middle = lower + ((upper - lower) >> 1);
        if (vec[middle] < search) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

__device__ __forceinline__
int2 search_2key(
    const int * __restrict__ arr,
    int left,
    int right,
    const int small_key,
    const int large_key
) {
    int middle;
    int2 out = make_int2(-1, -1);
    while (left <= right) {
        middle = left + ((right - left) >> 1);
        if (arr[middle] < small_key)  {
            left = middle+1;
        } else if (arr[middle] == small_key) {
            out.x = middle;
            out.y = search(arr, middle+1, right, large_key);
            break;
        } else if (arr[middle] > small_key && arr[middle] < large_key) {
            if (left <= middle) {
                out.x = search(arr, left, middle, small_key);
            } else {
                out.x = middle;
            }

            if (middle+1 <= right) {
                out.y = search(arr, middle+1, right+1, large_key);
            } else {
                out.y = middle+1;
            }
            break;
        } else if (arr[middle] == large_key) {
            out.y = middle;
            out.x = search(arr, left, middle-1, small_key);
            break;
        } else if (arr[middle] > large_key) {
            right = middle-1;
        }
    }
    return out;
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
    const int wi = getId();
    if (wi >= numels) return;

    const int global_size = blockDim.x*gridDim.x;

    // se non riusciamo a prendere i due valori non entreremo mai perché la griglia dovrebbe essere lanciata diversamente
    for (int i = wi; i < numels-global_size; i += 2*global_size) {
        int curr = i;
        int succ = i + global_size;

        int el1 = v1[curr];
        int el2 = v1[succ];
        int2 index_el1_in_v2 = search_2key(v2, 0, numels-1, el1, el2);
        vmerge[curr + index_el1_in_v2.x] = el1;
        vmerge[succ + index_el1_in_v2.y] = el2;

        int el1_2 = v2[curr];
        int el2_2 = v2[succ];
        int2 index_el2_in_v1 = search_2key(v1, 0, numels-1, el1_2, el2_2);
        vmerge[curr + index_el2_in_v1.x] = el1_2;
        vmerge[succ + index_el2_in_v1.y] = el2_2;
    }

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
