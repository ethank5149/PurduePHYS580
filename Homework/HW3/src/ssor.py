from numba import jit
from numpy import diag, tril, triu, dot
from numpy.linalg import norm


@jit
def ssor(A, b, x, w=1, max_iter=500):
    D = diag(diag(A))
    R = A - D
    L, U = tril(R), triu(R)
    M = (1 / (w * (2 - w))) * dot(dot(D + w * L, inv(D)), (D + w * U))
    N = M - A
    M_inv = inv(M)
    M_inv_dot_N = dot(M_inv, N)
    M_inv_dot_b = dot(M_inv, b)

    for i in range(max_iter):
        x = dot(M_inv_dot_N, x) + M_inv_dot_b
    return x