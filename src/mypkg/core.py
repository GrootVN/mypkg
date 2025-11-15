import numpy as np
import matplotlib.pyplot as plt
import numba
import heapq

@numba.jit(cache=True)
def fastVAT(
    matrix_of_pairwise_distance: np.ndarray, inplace=False
) -> tuple[np.ndarray, list[int]]:
    N = matrix_of_pairwise_distance.shape[0]
    if inplace:
        ordered_matrix = matrix_of_pairwise_distance
    else:
        ordered_matrix: np.ndarray = np.zeros(matrix_of_pairwise_distance.shape)
    p: list[int] = vat_prim_mst(matrix_of_pairwise_distance)
    # Step 3 - since this is symmetric, we only have to do half
    n_bit_mask = int(np.ceil(N / 8))
    visited = np.zeros(
        (N, n_bit_mask), dtype=np.uint8
    )  # Boolean is stored as a byte, so this is smaller

    for ij in range(N):
        for jk in range(ij, N):
            if not inplace:
                ordered_matrix[ij, jk] = ordered_matrix[jk, ij] = (
                    matrix_of_pairwise_distance[p[ij], p[jk]]
                )
            else:
                if get_bit(visited, ij, jk):
                    continue
                # Walk this loop, and store which visited
                r0, c0 = ij, jk
                r1, c1 = -1, -1
                p0 = ordered_matrix[r0, c0]
                while r1 != ij or c1 != jk:
                    r1, c1 = p[r0], p[c0]
                    set_bit(visited, r0, c0)
                    set_bit(visited, c0, r0)
                    ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = ordered_matrix[
                        r1, c1
                    ]
                    # Next step!
                    r0, c0 = r1, c1
                # Close the final block
                ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = p0
                set_bit(visited, r0, c0)
                set_bit(visited, c0, r0)

    # Step 4 - since this is symmetric, we only have to do half
    return ordered_matrix, p


@numba.jit(cache=True)
def set_bit(bitmask, row, col):
    bitmask[row, col // 8] |= 1 << (col % 8)


@numba.jit(cache=True)
def get_bit(bitmask, row, col):
    return (bitmask[row, col // 8] >> (col % 8)) & 1


@numba.jit(cache=True)
def vat_prim_mst(adj: np.ndarray) -> np.ndarray:
    N = len(adj)

    # Find the column of the maximum value.
    max_adj = np.argmax(adj)
    src = max_adj // N
    src_key = np.max(adj)

    # Create a list for keys and initialize all keys as infinite (INF)
    key: np.ndarray = np.full(N, float("inf"))

    # To store the parent array which, in turn, stores MST
    parent: np.ndarray = np.full(N, -1)

    # To keep track of vertices included in MST
    in_mst = np.full(N, False)

    # Insert the source itself into the priority queue and initialize its key as 0
    pq: list[tuple[float, int]] = [
        (src_key, src)
    ]  # Priority queue to store vertices that are being processed
    key[src] = src_key

    # The final sequence of vertices in MST
    heap_seq: np.ndarray = np.zeros(N, dtype=np.int32)
    heap_seq_idx = 0

    # Loop until the priority queue becomes empty
    while pq:
        # The first vertex in the pair is the minimum key vertex
        # Extract it from the priority queue
        # The vertex label is stored in the second of the pair
        u = heapq.heappop(pq)[1]

        # Different key values for the same vertex may exist in the priority queue.
        # The one with the least key value is always processed first.
        # Therefore, ignore the rest.
        if in_mst[u]:
            continue

        in_mst[u] = True  # Include the vertex in MST
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        # Iterate through all adjacent vertices of a vertex
        # Parallel processing of adjacent vertices
        vertices = np.arange(N)
        mask = (vertices != u) & ~in_mst & (key[vertices] > adj[u, vertices])
        key[mask] = adj[u, mask]
        for v in vertices[mask]:
            heapq.heappush(pq, (key[v], v))
            parent[v] = u

    return heap_seq

def VAT(R):
    """
    VAT algorithm.

    Parameters:
    R (ndarray): Dissimilarity data input.

    Returns:
    RV (ndarray): VAT-reordered dissimilarity data.
    C (ndarray): Connection indexes of MST.
    I (ndarray): Reordered indexes of R, the input data.
    RI (ndarray): Reordered indexes in the original data.
    """

    N, M = R.shape
    K = np.arange(N)
    J = K.copy()
    i = np.argmax(R, axis=0)
    y = np.max(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)

    I = i[j]
    J = np.delete(J,I)
    y = np.min(R[I,J])
    j = np.argmin(R[I,J])

    I = np.array([I, J[j]])
    J = np.delete(J, np.where(J == J[j]))
    C = np.array([0, 0])

    for r in range(2, N-1):

        submatrix = R[np.ix_(I, J)]
        y = np.min(submatrix, axis=0); i = np.argmin(submatrix, axis=0)
        y, j = np.min(y), np.argmin(y)
        I = np.append(I, J[j])
        J = np.delete(J, np.where(J == J[j]))
        C = np.append(C, i[j])
    
    submatrix = R[np.ix_(I, J)]
    y = np.min(submatrix, axis=0); i = np.argmin(submatrix, axis=0)
    I = np.append(I, J)
    C = np.append(C, i)

    RI = np.arange(N)
    for r in range(0, N):
        RI[I[r]] = r
    RV = R[np.ix_(I, I)]
    return RV, C, I, RI


@numba.jit(cache=True)
def iVAT(R, VATflag=False, fastVATflag=True, cutflag=False):
    N, M = R.shape

    if N != M:
        raise ValueError("R should be a square matrix")

    # if VATflag:
    #     RV = R
    #     RiV = np.zeros((N, N))
    #     for r in range(1, N):
    #         c = list(range(r))
    #         y, i = min(RV[r, :r]), np.argmin(RV[r, :r])
    #         RiV[r, c] = y
    #         cnei = [ci for ci in c if ci != i]
    #         RiV[r, cnei] = np.max([RiV[r, cnei], RiV[i, cnei]], axis=0)
    #         RiV[c, r] = RiV[r, c]
    # else:
    #     RV, C, I, RI = VAT(R)
    #     RiV = np.zeros((N, N))
    #     for r in range(1, N):
    #         c = list(range(r))
    #         RiV[r, c] = RV[r, C[r-1]]
    #         cnei = [ci for ci in c if ci != C[r-1]]
    #         RiV[r, cnei] = np.max([RiV[r, cnei], RiV[C[r-1], cnei]], axis=0)
    #         RiV[c, r] = RiV[r, c]

    if not VATflag: # R is original dissimilarity matrix (run VAT)
        if fastVATflag:
            RV,_ = fastVAT(R)
        else:
            RV,_,_,_ = VAT(R)
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = list(range(r))
            y, i = min(RV[r, :r]), np.argmin(RV[r, :r])
            RiV[r, c] = y
            cnei = [ci for ci in c if ci != i]
            RiV[r, cnei] = np.max([RiV[r, cnei], RiV[i, cnei]], axis=0)
            RiV[c, r] = RiV[r, c]
    else:   # R is ordered matrix from VAT or fastVAT
        RV = R
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = list(range(r))
            y, i = min(RV[r, :r]), np.argmin(RV[r, :r])
            RiV[r, c] = y
            cnei = [ci for ci in c if ci != i]
            RiV[r, cnei] = np.max([RiV[r, cnei], RiV[i, cnei]], axis=0)
            RiV[c, r] = RiV[r, c]

    if cutflag:
        plt.imshow(RiV, cmap='gray', aspect='auto')
        plt.title('Click on the darkest value between blocks to select the cut distance')
        plt.show()
        cutval = float(input("Enter the cut value: "))
        C = np.zeros(N, dtype=int)
        ind = list(range(N))
        i = [idx for idx, val in enumerate(np.diag(RiV, 1)) if val >= cutval]
        for j in range(len(i) - 1):
            C[ind < i[j+1]] = j + 1
        C[ind > i[-1]] = len(i)
    else:
        C = []

    return RiV, RV
