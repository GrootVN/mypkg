import numpy as np
import matplotlib.pyplot as plt
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



def iVAT(R, VATflag=False, cutflag=False):
    N, M = R.shape

    if N != M:
        raise ValueError("R should be a square matrix")

    if VATflag:
        RV = R
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = list(range(r))
            y, i = min(RV[r, :r]), np.argmin(RV[r, :r])
            RiV[r, c] = y
            cnei = [ci for ci in c if ci != i]
            RiV[r, cnei] = np.max([RiV[r, cnei], RiV[i, cnei]], axis=0)
            RiV[c, r] = RiV[r, c]
    else:
        RV, C, I, RI = VAT(R)
        RiV = np.zeros((N, N))
        for r in range(1, N):
            c = list(range(r))
            RiV[r, c] = RV[r, C[r-1]]
            cnei = [ci for ci in c if ci != C[r-1]]
            RiV[r, cnei] = np.max([RiV[r, cnei], RiV[C[r-1], cnei]], axis=0)
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