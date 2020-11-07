from numpy import asarray, sin, cos, pi, cross, sqrt
from numpy.linalg import norm
from scipy.integrate import quad
from functools import partial

def biot_savart(pos, path, dpath, I):
    x = quad(partial(lambda s, pos, path, dpath, I : I * ((pos[2] - path(s)[2]) * dpath(s)[1] - (pos[1] - path(s)[1]) * dpath(s)[2]) / norm(pos - path(s)) ** 3, pos=pos, path=path, dpath=dpath, I=I), 0, 1)[0]
    y = quad(partial(lambda s, pos, path, dpath, I : I * ((pos[0] - path(s)[0]) * dpath(s)[2] - (pos[2] - path(s)[2]) * dpath(s)[0]) / norm(pos - path(s)) ** 3, pos=pos, path=path, dpath=dpath, I=I), 0, 1)[0]
    z = quad(partial(lambda s, pos, path, dpath, I : I * ((pos[1] - path(s)[1]) * dpath(s)[0] - (pos[0] - path(s)[0]) * dpath(s)[1]) / norm(pos - path(s)) ** 3, pos=pos, path=path, dpath=dpath, I=I), 0, 1)[0]
    return asarray([x, y, z])

def B_solenoid_z(pos, radius, l, n, I):
    return quad(lambda s : 2 * pi * n * I * radius * (radius - pos[0] * cos(2 * pi * n * s) - pos[1] * sin(2 * pi * n * s)) / ((pos[2] + l * (1 / 2 - s)) ** 2 + (pos[0] - radius * cos(2 * n * pi * s)) ** 2 + (pos[1] - radius * sin(2 * n * pi * s)) ** 2) ** (3 / 2), 0, 1)[0]
    
def B_solenoid_z_analytic(z, radius, l, n, I):
    return 2 * pi * n * I * ((l / 2 + z) / sqrt((l / 2 + z) ** 2 + radius ** 2) + (l / 2 - z) / sqrt((l / 2 - z) ** 2 + radius ** 2))