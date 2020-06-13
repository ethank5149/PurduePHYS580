from typing import Union, List, Tuple
import numpy as np
from numpy.core._multiarray_umath import ndarray
from lib.FindRoot import bisection, secant


# TODO: Implement symplectic integrators
# TODO: Maybe re-implement rkf, rkf45, etc.?


def euler(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
          terminate: callable = lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with Euler's method.

    Usage:
        x = euler(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].
    """

    n = np.size(t)
    x = np.zeros((n,np.size(x0)))
    x[0] = np.asarray(x0)
    # x = np.asarray([x0] * n, dtype="float")

    for i in range(n - 1):
        x[i + 1] = x[i] + (t[i + 1] - t[i]) * f(t[i], x[i])
        if terminate(x[i]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def heun(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with Heun's method.

    USAGE:
        x = heun(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].
    """

    n = np.size(t)
    x = np.array([x0] * n, dtype="float")
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i + 1], x[i] + k1)
        x[i + 1] = x[i] + (k1 + k2) / 2
        if terminate(x[i]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk2a(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with the second-order Runge-Kutta method.

    USAGE:
        x = rk2a(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].

    """

    n = np.size(t)
    x = np.array([x0] * n, dtype="float")
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], x[i]) / 2
        x[i + 1] = x[i] + h * f(t[i] + h / 2, x[i] + k1)
        if terminate(x[i]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk2b(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with the second-order Runge-Kutta method.

    USAGE:
        x = rk2b(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].

    """

    n = np.size(t)
    x = np.array([x0] * n, dtype="float")
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i + 1], x[i] + k1)
        x[i + 1] = x[i] + (k1 + k2) / 2
        if terminate(x[i]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk4(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
        terminate: callable = lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with the fourth-order Runge-Kutta method.

    USAGE:
        x = rk4(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s) x(t[0]).
        t     - t values to compute solution at.
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].
    """

    n = np.size(t)
    x = np.array([x0] * n, dtype="float")
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i] + 0.5 * h, x[i] + 0.5 * k1)
        k3 = h * f(t[i] + 0.5 * h, x[i] + 0.5 * k2)
        k4 = h * f(t[i + 1], x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2 * (k2 + k3) + k4) / 6
        if terminate(x[i]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def shoot(f: callable, a: float, b: float, c1: float, c2: float, t: Union[List, ndarray],
          boundary_type: str = 'dirichlet') -> ndarray:
    """Implements the shooting method to solve second order BVPs

    Usage:
        y = shoot(f, a, b, c1, c2, t)

    Input:
        f       - f(t,y) = dy/dt such that f: [x0,x1] -> [x0',x1'].
        a       - a = y(t[0]).
        b       - b = y(t[n-1]).
        c1      - first initial estimate of y'(t[0]).
        c1      - second initial estimate of y'(t[0]).
        t       - array of n time values to determine y at.

    Return:
        y       - solution for each t[i].

    """

    if boundary_type == 'dirichlet':
        def f_comp(c):
            y_c = rk4(f, [a, c], t)
            return y_c[0, -1] - b
    else:  # boundary_type == 'neumann':
        def f_comp(c):
            y_c = rk4(f, [a, c], t)
            return y_c[1, -1] - b

    c_opt = bisection(f_comp, c1, c2)
    return rk4(f, [a, c_opt], t)


def shoot_angle(f: callable, pos: Tuple, v0: float, a: float, b: float, c1: float, c2: float, t: Union[List, ndarray],
          boundary_type: str = 'dirichlet') -> ndarray:
    """Implements the shooting method to solve second order BVPs

    Usage:
        y = shoot(f, a, b, c1, c2, t)

    Input:
        f       - f(t,y) = dy/dt such that f: [x0,x1] -> [x0',x1'].
        a       - a = y(t[0]).
        b       - b = y(t[n-1]).
        c1      - first initial estimate of y'(t[0]).
        c1      - second initial estimate of y'(t[0]).
        t       - array of n time values to determine y at.

    Return:
        y       - solution for each t[i].

    """

    if boundary_type == 'dirichlet':
        def f_comp(angle):
            y_c = rk4(f, [*pos, v0 * np.cos(np.pi*angle/180), v0 * np.sin(np.pi*angle/180)], t)
            return y_c[0, -1] - b
    else:  # boundary_type == 'neumann':
        def f_comp(angle):
            y_c = rk4(f, [*pos, v0 * np.cos(np.pi*angle/180), v0 * np.sin(np.pi*angle/180)], t)
            return y_c[1, -1] - b

    c_opt = bisection(f_comp, c1, c2)
    return rk4(f, [a, c_opt], t)


def shoot_projectile_angle(f: callable, bc: callable, ic: Tuple, root_params: Tuple,
                           t: Union[List, ndarray]) -> ndarray:
    """Implements the shooting method to solve second order BVPs

    Usage:
        y = shoot(f, a, b, c1, c2, t)

    Input:
        f       - f(t,y) = dy/dt such that f: [x0,x1] -> [x0',x1'].
        a       - a = y(t[0]).
        b       - b = y(t[n-1]).
        c1      - first initial estimate of y'(t[0]).
        c1      - second initial estimate of y'(t[0]).
        t       - array of n time values to determine y at.

    Return:
        y       - solution for each t[i].

    """


    def terminate(X):
        return X[1] < 0


    def f_comp(angle):
        y_angle = rk4(f, [ic[0], ic[1], ic[2] * np.cos(angle), ic[2] * np.sin(angle)], t, terminate=terminate)
        return bc(y_angle)
    if len(root_params) == 2:
        root_params = (*root_params, None)
    angle_opt = bisection(f_comp, root_params[0], root_params[1],tol=root_params[2])
    return angle_opt
#    return rk4(f, [ic[0], ic[1], ic[2] * np.cos(angle_opt), ic[2] * np.sin(angle_opt)], t)
