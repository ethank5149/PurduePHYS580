from numba import jit
from numpy import diag, tril, triu, dot
from numpy.linalg import inv, norm


@jit
def sor(A, b, x, w=1, max_iter=1e3):
    D = diag(diag(A))
    R = A - D
    L, U = tril(R), triu(R)
    M = D / w + L
    N = M - A
    M_inv = inv(M)
    M_inv_dot_N = dot(M_inv, N)
    M_inv_dot_b = dot(M_inv, b)

    for i in range(max_iter):
        x = dot(M_inv_dot_N, x) + M_inv_dot_b
    return x


@jit
def sor_e(A, b, x, w=1, max_iter=1e9, min_err=1):
    D = diag(diag(A))
    R = A - D
    L, U = tril(R), triu(R)
    M = D / w + L
    N = M - A
    M_inv = inv(M)
    M_inv_dot_N = dot(M_inv, N)
    M_inv_dot_b = dot(M_inv, b)

    for i in range(max_iter):
        _ = dot(M_inv_dot_N, x) + M_inv_dot_b
        if norm(_ - x) < min_err:
            return _, i
        else:
            x = _
    return x, -1