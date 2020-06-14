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


def eulercromer(f, x0, dx0, t, terminate=lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with the Euler-Cromer method.

    Usage:
        x = eulercromer(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].
        dx     - Solution derivative values at each t[i].
    """

    n = np.size(t)
    x = np.zeros((n,np.size(x0)))
    dx = np.zeros((n,np.size(dx0)))

    x[0] = np.asarray(x0)
    dx[0] = np.asarray(dx0)

    for i in range(n - 1):
        dx[i + 1] = dx[i] + (t[i + 1] - t[i]) * f(t[i], x[i], dx[i])
        x[i + 1] = x[i] + (t[i + 1] - t[i]) * dx[i+1]
        if terminate(x[i], dx[i]):
            dx = dx[:i + 1]
            x = x[:i+1]
            t = t[:i+1]
            break

    return x.T, dx.T


def eulercromer_pendulum(f, x0, dx0, t, terminate=lambda *args: False) -> ndarray:
    """Solves x' = f(t,x) with the Euler-Cromer method.

    Usage:
        x = eulercromer(f, x0, t)

    Input:
        f     - f(t,x) = dx/dt.
        x0    - The initial condition(s), x(t[0]).
        t     - t values to compute solution at. The difference h=t[i+1]-t[i]
                determines the step size h.

    Returns:
        x     - Solution values at each t[i].
        dx     - Solution derivative values at each t[i].
    """

    n = np.size(t)
    x = np.zeros((n,np.size(x0)))
    dx = np.zeros((n,np.size(dx0)))

    x[0] = np.asarray(x0)
    dx[0] = np.asarray(dx0)

    for i in range(n - 1):
        dx[i + 1] = dx[i] + (t[i + 1] - t[i]) * f(t[i], x[i], dx[i])
        x[i + 1] = x[i] + (t[i + 1] - t[i]) * dx[i+1]

        if x[i,0]<-np.pi:
            x[i,0] += 2*np.pi
        elif x[i, 0] > np.pi:
            x[i, 0] -= 2 * np.pi
    return x.T, dx.T


def rk4_pendulum(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List]) -> ndarray:
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

        if x[i,0]<-np.pi:
            x[i,0] += 2*np.pi
        elif x[i, 0] > np.pi:
            x[i, 0] -= 2 * np.pi
    return x.T