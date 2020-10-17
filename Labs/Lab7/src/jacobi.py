from numba import jit
import numpy as np

@jit
def jacobi(A, b, x, max_iter=500):
    D = np.diag(A)
    R = A - np.diag(D)
    for i in range(max_iter):
        x = (b - np.dot(R, x)) / D
    return x