from numba import jit
from numpy import diag, dot
from numpy.linalg import norm


@jit
def jacobi(A, b, x, max_iter=1e3):
    D = diag(A)
    R = A - diag(D)
    for i in range(max_iter):
        x = (b - dot(R, x)) / D
    return x


@jit
def jacobi_e(A, b, x, max_iter=1e9, min_err=1):
    D = diag(A)
    R = A - diag(D)
    for i in range(max_iter):
        _ = (b - dot(R, x)) / D
        if norm(_ - x) < min_err:
            return _, i
        else:
            x = _
    return x, -1