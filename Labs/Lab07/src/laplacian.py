from numba import jit
import numpy as np


@jit
def laplacian_1d(n):
    return np.diag(2 * np.ones(n - 2)) + \
           np.diag(-np.ones(n - 3), 1) + \
           np.diag(-np.ones(n - 3), -1)
           

@jit
def laplacian_2d(n):
    return np.kron(np.eye(n - 2), laplacian_1d(n)) + \
           np.kron(laplacian_1d(n), np.eye(n - 2))



# from scipy.sparse import diags, kron, eye

# @jit
# def laplacian_1d(n):
#     return diags([2 * np.ones(n - 2), -np.ones(n - 3), -np.ones(n - 3)], [0, 1, -1])
           

# @jit
# def laplacian_2d(n):
#     return kron(eye(n - 2), laplacian_1d(n)) + \
#            kron(laplacian_1d(n), eye(n - 2))