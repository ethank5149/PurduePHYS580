from numba import jit
from numpy import diag, ones, kron, eye


@jit
def laplacian_1d(n):
    return diag(-2 * ones(n - 2)) + \
           diag(ones(n - 3), 1) + \
           diag(ones(n - 3), -1)
           

@jit
def laplacian_2d(n):
    return kron(eye(n - 2), laplacian_1d(n)) + \
           kron(laplacian_1d(n), eye(n - 2))