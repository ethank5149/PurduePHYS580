from numba import jit
from numpy import diag, tril, triu, dot
from numpy.linalg import inv


@jit
def gauss_seidel(A, b, x, w=1, max_iter=1e3):
    D = diag(diag(A))
    R = A - D
    L, U = tril(R), triu(R)
    M = D + L
    N = M - A
    
    M_inv = inv(M)
    M_inv_dot_N = dot(M_inv, N)
    M_inv_dot_b = dot(M_inv, b)

    for i in range(max_iter):
        x = dot(M_inv_dot_N, x) + M_inv_dot_b
    return x