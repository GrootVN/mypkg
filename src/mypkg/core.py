import numpy as np
import matplotlib.pyplot as plt
import numba
import heapq


@numba.jit(cache=True)
def fastVAT(
    matrix_of_pairwise_distance: np.ndarray, inplace: bool = False
) -> tuple[np.ndarray, list[int]]:
    """
    Fast VAT (Visual Assessment of cluster Tendency) reordering using an MST.

    This is an optimized VAT implementation that:
    - Computes a permutation of indices `p` using a Prim-style MST on the
      dissimilarity matrix.
    - Reorders the input matrix according to `p` to produce a VAT-ordered
      dissimilarity matrix.

    Parameters
    ----------
    matrix_of_pairwise_distance : np.ndarray, shape (N, N)
        Symmetric pairwise dissimilarity (distance) matrix.
    inplace : bool, default=False
        If False:
            A new reordered matrix is allocated and returned.
        If True:
            Reordering is (attempted to be) performed in-place on
            `matrix_of_pairwise_distance`. This uses a bitmask to avoid
            repeatedly touching already processed entries.

    Returns
    -------
    ordered_matrix : np.ndarray, shape (N, N)
        VAT-reordered dissimilarity matrix.
    p : list[int]
        Reordering (permutation) of indices. Row/column i in `ordered_matrix`
        corresponds to original row/column p[i] in `matrix_of_pairwise_distance`.
    """
    N = matrix_of_pairwise_distance.shape[0]

    if inplace:
        ordered_matrix = matrix_of_pairwise_distance
    else:
        ordered_matrix = np.zeros(matrix_of_pairwise_distance.shape)

    # Step 1–2: obtain VAT ordering via MST
    p: list[int] = vat_prim_mst(matrix_of_pairwise_distance)

    # Step 3 – since this is symmetric, we only have to do half
    n_bit_mask = int(np.ceil(N / 8))
    # Boolean is stored as a byte, so this is smaller than a full boolean matrix
    visited = np.zeros((N, n_bit_mask), dtype=np.uint8)

    for ij in range(N):
        for jk in range(ij, N):
            if not inplace:
                # Simple copy from reordered indices
                ordered_matrix[ij, jk] = ordered_matrix[jk, ij] = (
                    matrix_of_pairwise_distance[p[ij], p[jk]]
                )
            else:
                # In-place rewriting, skipping cells we've already handled
                if get_bit(visited, ij, jk):
                    continue

                # Walk this loop, and store which have been visited
                r0, c0 = ij, jk
                r1, c1 = -1, -1
                p0 = ordered_matrix[r0, c0]  # keep original value

                while r1 != ij or c1 != jk:
                    r1, c1 = p[r0], p[c0]
                    set_bit(visited, r0, c0)
                    set_bit(visited, c0, r0)
                    ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = ordered_matrix[
                        r1, c1
                    ]
                    # Next step in the cycle
                    r0, c0 = r1, c1

                # Close the final block
                ordered_matrix[r0, c0] = ordered_matrix[c0, r0] = p0
                set_bit(visited, r0, c0)
                set_bit(visited, c0, r0)

    # Step 4 – symmetric matrix, so we only needed half
    return ordered_matrix, p


@numba.jit(cache=True)
def set_bit(bitmask: np.ndarray, row: int, col: int) -> None:
    """
    Set a bit in a compact 2D bitmask.

    The bitmask is stored as (N, ceil(N/8)) bytes, where each byte holds 8 bits.

    Parameters
    ----------
    bitmask : np.ndarray, shape (N, ceil(N/8))
        Byte-level bitmask.
    row : int
        Row index.
    col : int
        Column index.
    """
    bitmask[row, col // 8] |= 1 << (col % 8)


@numba.jit(cache=True)
def get_bit(bitmask: np.ndarray, row: int, col: int) -> int:
    """
    Get the value of a bit in a compact 2D bitmask.

    Parameters
    ----------
    bitmask : np.ndarray, shape (N, ceil(N/8))
        Byte-level bitmask.
    row : int
        Row index.
    col : int
        Column index.

    Returns
    -------
    int
        1 if the bit is set, 0 otherwise.
    """
    return (bitmask[row, col // 8] >> (col % 8)) & 1


@numba.jit(cache=True)
def vat_prim_mst(adj: np.ndarray) -> np.ndarray:
    """
    Prim-style MST-based ordering for VAT.

    Given a dissimilarity (adjacency) matrix, this computes an ordering
    of vertices by:
    - Selecting the vertex corresponding to the maximum distance as a seed
    - Building an MST using a min-priority queue
    - Recording the sequence in which vertices are added

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Symmetric pairwise dissimilarity matrix.

    Returns
    -------
    heap_seq : np.ndarray, shape (N,)
        Sequence of vertex indices describing the VAT ordering.
    """
    N = len(adj)

    # Find the column of the maximum value.
    max_adj = np.argmax(adj)
    src = max_adj // N
    src_key = np.max(adj)

    # Keys: best known edge weight connecting each vertex to MST
    key = np.full(N, float("inf"))

    # Parents in MST (not used here for reordering, but kept for completeness)
    parent = np.full(N, -1)

    # Whether a vertex is in MST
    in_mst = np.full(N, False)

    # Priority queue of (key, vertex)
    pq: list[tuple[float, int]] = [(src_key, src)]
    key[src] = src_key

    # Final MST visit sequence
    heap_seq = np.zeros(N, dtype=np.int32)
    heap_seq_idx = 0

    while pq:
        # Pop vertex with smallest key
        u = heapq.heappop(pq)[1]

        # Skip if already processed (stale queue entries)
        if in_mst[u]:
            continue

        in_mst[u] = True
        heap_seq[heap_seq_idx] = u
        heap_seq_idx += 1

        # Update neighbors
        vertices = np.arange(N)
        mask = (vertices != u) & ~in_mst & (key[vertices] > adj[u, vertices])
        key[mask] = adj[u, mask]
        for v in vertices[mask]:
            heapq.heappush(pq, (key[v], v))
            parent[v] = u

    return heap_seq


def VAT(R: np.ndarray):
    """
    VAT (Visual Assessment of cluster Tendency) algorithm.

    This is the classic VAT implementation that:
    - Reorders a dissimilarity matrix R using a greedy strategy
      (not the MST-based fastVAT).
    - Produces a re-ordered matrix RV whose image can reveal
      cluster structure via darker blocks along the diagonal.

    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Dissimilarity (distance) matrix.

    Returns
    -------
    RV : np.ndarray, shape (N, N)
        VAT-reordered dissimilarity matrix.
    C : np.ndarray, shape (N,)
        Connection indexes / parents (MST-like).
    I : np.ndarray, shape (N,)
        Reordered indices of R (permutation).
    RI : np.ndarray, shape (N,)
        Reverse mapping: RI[original_index] = VAT position.
    """
    N, M = R.shape
    K = np.arange(N)
    J = K.copy()

    # 1. Find the object with maximum distance
    i = np.argmax(R, axis=0)
    y = np.max(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)

    I = i[j]
    J = np.delete(J, I)

    # 2. Add the next closest object
    y = np.min(R[I, J])
    j = np.argmin(R[I, J])
    I = np.array([I, J[j]])
    J = np.delete(J, np.where(J == J[j]))
    C = np.array([0, 0])

    # 3. Greedily add remaining objects
    for r in range(2, N - 1):
        submatrix = R[np.ix_(I, J)]
        y = np.min(submatrix, axis=0)
        i = np.argmin(submatrix, axis=0)
        y, j = np.min(y), np.argmin(y)
        I = np.append(I, J[j])
        J = np.delete(J, np.where(J == J[j]))
        C = np.append(C, i[j])

    # 4. Add final object
    submatrix = R[np.ix_(I, J)]
    y = np.min(submatrix, axis=0)
    i = np.argmin(submatrix, axis=0)
    I = np.append(I, J)
    C = np.append(C, i)

    # 5. Build inverse permutation
    RI = np.arange(N)
    for r in range(0, N):
        RI[I[r]] = r

    # 6. Reorder matrix
    RV = R[np.ix_(I, I)]
    return RV, C, I, RI


@numba.jit(cache=True)
def iVAT(R: np.ndarray, VATflag: bool = False, fastVATflag: bool = True, cutflag: bool = False):
    """
    iVAT (improved VAT) algorithm.

    iVAT sharpens cluster structure by transforming the VAT-ordered
    dissimilarity matrix into a graph-based distance that emphasizes
    path-connectedness (via MST-based distances).

    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Input matrix. Its meaning depends on `VATflag`:
        - If VATflag is False: R is the original dissimilarity matrix.
        - If VATflag is True : R is assumed already VAT-ordered (RV).
    VATflag : bool, default=False
        If False:
            Compute VAT reordering (using fastVAT or VAT) internally first.
        If True:
            Treat R as already VAT-ordered, skip VAT/fastVAT step.
    fastVATflag : bool, default=True
        If VATflag is False:
            If True, use `fastVAT` to compute VAT ordering.
            If False, use the classic `VAT` function.
    cutflag : bool, default=False
        If True:
            Display `RiV` as an image and interactively ask for a cut
            distance to derive cluster labels C.
        If False:
            No interactive cutting is performed and C is returned as an
            empty list.

    Returns
    -------
    RiV : np.ndarray, shape (N, N)
        iVAT-transformed dissimilarity matrix, showing sharper block structure.
    RV : np.ndarray, shape (N, N)
        VAT-ordered dissimilarity matrix used to create RiV.

    Notes
    -----
    - If `cutflag` is True, the user is asked to choose a cut distance
      based on the image of `RiV`. Cluster label vector `C` is computed
      but not returned by this function in the current implementation.
    """
    N, M = R.shape

    if N != M:
        raise ValueError("R should be a square matrix")

    if not VATflag:
        # R is original dissimilarity matrix (run VAT)
        if fastVATflag:
            print("Using fastVAT...")
            RV, _ = fastVAT(R)
        else:
            print("Using VAT...")
            RV, _, _, _ = VAT(R)
        print("VAT reordering completed.")

        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = list(range(r))
            # Direct neighbor with minimal distance in VAT-ordered space
            y, i = min(RV[r, :r]), np.argmin(RV[r, :r])
            RiV[r, c] = y
            # Other neighbors (path-based max distance)
            cnei = [ci for ci in c if ci != i]
            RiV[r, cnei] = np.max([RiV[r, cnei], RiV[i, cnei]], axis=0)
            RiV[c, r] = RiV[r, c]
    else:
        # R is ordered matrix from VAT or fastVAT
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
        # Interactive cut selection (not returned from function)
        plt.imshow(RiV, cmap='gray', aspect='auto')
        plt.title('Click on the darkest value between blocks to select the cut distance')
        plt.show()
        cutval = float(input("Enter the cut value: "))

        C = np.zeros(N, dtype=int)
        ind = list(range(N))
        # Off-diagonal values (k=1) used to determine cut positions
        i = [idx for idx, val in enumerate(np.diag(RiV, 1)) if val >= cutval]
        for j in range(len(i) - 1):
            C[ind < i[j + 1]] = j + 1
        C[ind > i[-1]] = len(i)
        # NOTE: Cluster labels C are not returned in current implementation.

    return RiV, RV
