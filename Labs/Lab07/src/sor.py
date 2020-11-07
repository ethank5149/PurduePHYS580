from numba import jit
import numpy as np


@jit
def sor(A, b, x, w=1, max_iter=500):
    D = np.diag(np.diag(A))
    R = A - D
    L, U = np.tril(R), np.triu(R)
    M = D / w + L
    N = M - A
    M_inv = np.linalg.inv(M)
    M_inv_dot_N = np.dot(M_inv, N)
    M_inv_dot_b = np.dot(M_inv, b)

    for i in range(max_iter):
        x = np.dot(M_inv_dot_N, x) + M_inv_dot_b
    return x


# from scipy.sparse import diags, tril, triu
# from scipy.sparse.linalg import inv

# @jit
# def sor(A, b, x, w=1, max_iter=500):
#     D = diags(A.diagonal())
#     R = A - D
#     L, U = tril(R), triu(R)
#     M = D / w + L
#     N = M - A
#     M_inv = inv(M)
#     M_inv_dot_N = M_inv.dot(N)
#     M_inv_dot_b = M_inv.dot(b)

#     for i in range(max_iter):
#         x = M_inv_dot_N.dot(x) + M_inv_dot_b
#     return x