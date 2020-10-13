from numpy import hsplit, hstack
from numpy.linalg import norm

def rhs(t, y, mu, k, L, G):
    m1, m2, m3, m4 = mu
    x1, x2, x3, x4, v1, v2, v3, v4 = hsplit(y, 8)
    return hstack((
        v1,
        v2,
        v3,
        v4,
        -G * (m2 * (x1 - x2) / norm(x1 - x2) ** 3 + m3 * (x1 - x3) / norm(x1 - x3) ** 3 + m4 * (x1 - x4) / norm(x1 - x4) ** 3),
        -G * (m1 * (x2 - x1) / norm(x2 - x1) ** 3 + m3 * (x2 - x3) / norm(x2 - x3) ** 3 + m4 * (x2 - x4) / norm(x2 - x4) ** 3),
        -G * (m1 * (x3 - x1) / norm(x3 - x1) ** 3 + m2 * (x3 - x2) / norm(x3 - x2) ** 3 + m4 * (x3 - x4) / norm(x3 - x4) ** 3) + k * (norm(x4 - x3) - L) ** 2 * (x4 - x3) / norm(x4 - x3),
        -G * (m1 * (x4 - x1) / norm(x4 - x1) ** 3 + m2 * (x4 - x2) / norm(x4 - x2) ** 3 + m3 * (x4 - x3) / norm(x4 - x3) ** 3) + k * (norm(x3 - x4) - L) ** 2 * (x3 - x4) / norm(x3 - x4)
    ))