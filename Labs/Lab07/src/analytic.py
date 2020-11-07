# from numba import jit
import numpy as np


# @jit
def analytic(X, Y, N, H, L, V):
    output = np.zeros((X.size, Y.size))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            sum1 = np.sum([((-1) ** n - 1) * np.sin(n * np.pi * x / H) * np.sinh(n * np.pi * (L - y) / H) / (n * np.sinh(n * np.pi * L / H)) for n in range(1, N + 1)])
            sum2 = np.sum([((-1) ** n - 1) * np.sin(n * np.pi * x / H) * np.sinh(n * np.pi * y / H) / (n * np.sinh(n * np.pi * L / H)) for n in range(1, N + 1)])
            output[i, j] = -2 * V * (sum1 - sum2) / np.pi
    return output