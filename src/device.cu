__global__
void riduzione(
    const int2 * __restrict__ input,
    int * __restrict__ output,
    int search,
    int * idx,
    const int numels
) {
    int i = getId();
    if (i >= numels) return;

    // if (i != 0) return;

    int2 in = input[i];
    if (search > in.x && search < in.y) {
        *idx = i*2+1;
        output[i] = in.y;
    }
    if (search > in.y) {
        *idx = i*2+1;
        output[i] = in.y;
    }
    if (search < in.x) {
        *idx = i*2;
        output[i] = in.x;
    }
    printf("Search: %d, inx = %d, iny = %d, output[%d] = %d", search, in.x, in.y, i, output[i]);

    // output[i] = in.x + in.y;
}

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
    int el2 = v2[i];

    int index_el1_in_v2 = search(v2, el1, numels); // 2*i;
    int index_el2_in_v1 = search(v1, el2, numels); // i1+1;

    vmerge[i+index_el1_in_v2] = el1;
    vmerge[i+index_el2_in_v1] = el2;

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
