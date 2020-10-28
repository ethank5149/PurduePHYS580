from numpy import zeros, ones


def compose_soln(x, n, V):
    A = zeros((n, n))
    A[0, :] = V * ones(n)
    A[-1, :] = -V * ones(n)
    A[1 : -1, 1 : -1] = x.reshape((n - 2, n - 2))
    return A