__device__ __forceinline__
int2 search(
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
            if (search > curr.z) return make_int2(middle, 3);
            if (search > curr.y) return make_int2(middle, 2);
            if (search > curr.x) return make_int2(middle, 1);
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
    int2 val = make_int2((middle+lower), 0);

    if (curr.x != -1) {
        // caso in cui è l'ultimo valore della quartina
        if (search > curr.w) val = make_int2((middle+1), 0);
        if (search < curr.x) val = make_int2((middle), 0);
    }

    return val;
}

// TODO: refactoring dei parametri
__device__ __forceinline__
void place(
    const int2 position,
    const int i,
    const int intPos,
    int4 * __restrict__ vmerge,
    const int value
) {
    const int minIndex = position.y + intPos;
    const int index = i + position.x + (minIndex/4);
    const int internalIndex = module(minIndex, 4);
    if (internalIndex == 0) {
        vmerge[index].x = value;
        return;
    }
    if (internalIndex == 1) {
        vmerge[index].y = value;
        return;
    }
    if (internalIndex == 2) {
        vmerge[index].z = value;
        return;
    }
    if (internalIndex == 3) {
        vmerge[index].w = value;
        return;
    }
}

// TODO: refactoring dei parametri
__device__ __forceinline__
void searchAndPlace(
    const int i,
    const int4 elements,
    const int quarts,
    int4 * __restrict__ vmerge,
    const int4 * __restrict__ vec
) {
    int2 index_el_0_in_vec = search(vec, elements.x, quarts);
    place(index_el_0_in_vec, i, 0, vmerge, elements.x);

    int2 index_el_1_in_vec = search(vec, elements.y, quarts);
    place(index_el_1_in_vec, i, 1, vmerge, elements.y);

    int2 index_el_2_in_vec = search(vec, elements.z, quarts);
    place(index_el_2_in_vec, i, 2, vmerge, elements.z);

    int2 index_el_3_in_vec = search(vec, elements.w, quarts);
    place(index_el_3_in_vec, i, 3, vmerge, elements.w);
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
    int4 * __restrict__ vmerge,
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
    int quarts = numels/4;
    if (i >= quarts) return;

    int4 el1 = v1[i];
    searchAndPlace(i, el1, quarts, vmerge, v2);

    int4 el2 = v2[i];
    searchAndPlace(i, el2, quarts, vmerge, v1);

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
