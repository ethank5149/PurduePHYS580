import numpy as np

def compose_soln(x, n, V):
    A = np.zeros((n, n))
    A[0, :] = V * np.ones(n)
    A[-1, :] = -V * np.ones(n)
    A[1 : -1, 1 : -1] = x.reshape((n - 2, n - 2))
    return A