__device__ __forceinline__
int search(
    const int4 * __restrict__ vec,
    const int search,
    const int length
) {
    int lower = 0;
    int upper = length;
    int middle = 0;
    int4 curr = make_int4(-1, -1, -1, -1);
    while (lower < upper) {
        middle = (lower + upper) >> 1;
        curr = vec[middle];
        if (search >= curr.x && search < curr.w) {
            // cerca dentro
            if (search > curr.z) return middle*4 + 3;
            if (search > curr.y) return middle*4 + 2;
            if (search > curr.x) return middle*4 + 1;
        } else {
            if (search > curr.w) {
                // cerca sopra
                lower = middle + 1;
            }
            if (search < curr.x) {
                // cerca sotto
                upper = middle;
            }
        }
    }
    // caso in cui è il primo valore della quartina
    int val = (middle+lower)*4 + 0;

    if (curr.x != -1) {
        // caso in cui è l'ultimo valore della quartina
        if (search > curr.w) val = (middle+1)*4;
        if (search < curr.x) val = (middle)*4;
    }

    return val;
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
    const int4 * __restrict__ v1,
    const int4 * __restrict__ v2,
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
    if (i >= numels/4) return;

    int4 el1 = v1[i];
    int index_el1_0_in_v2 = search(v2, el1.x, numels/4);
    int index_el1_1_in_v2 = search(v2, el1.y, numels/4);
    int index_el1_2_in_v2 = search(v2, el1.z, numels/4);
    int index_el1_3_in_v2 = search(v2, el1.w, numels/4);

    vmerge[(i*4+0) + index_el1_0_in_v2] = el1.x;
    vmerge[(i*4+1) + index_el1_1_in_v2] = el1.y;
    vmerge[(i*4+2) + index_el1_2_in_v2] = el1.z;
    vmerge[(i*4+3) + index_el1_3_in_v2] = el1.w;

    int4 el2 = v2[i];
    int index_el2_0_in_v1 = search(v1, el2.x, numels/4);
    int index_el2_1_in_v1 = search(v1, el2.y, numels/4);
    int index_el2_2_in_v1 = search(v1, el2.z, numels/4);
    int index_el2_3_in_v1 = search(v1, el2.w, numels/4);

    vmerge[(i*4+0) + index_el2_0_in_v1] = el2.x;
    vmerge[(i*4+1) + index_el2_1_in_v1] = el2.y;
    vmerge[(i*4+2) + index_el2_2_in_v1] = el2.z;
    vmerge[(i*4+3) + index_el2_3_in_v1] = el2.w;

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
