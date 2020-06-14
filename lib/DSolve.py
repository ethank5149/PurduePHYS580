from typing import Union, List, Tuple
import numpy as np
from numpy.core._multiarray_umath import ndarray
# TODO: Maybe re-implement rkf, rkf45, etc.?

# Simplectic Integrator Coefficients
# 1st Order
c_1_1 = 1
d_1_1 = 1
# 2nd Order
c_2_1, c_2_2 = 0, 1
d_2_1, d_2_2 = 0.5, 0.5
# Third Order
c_3_1, c_3_2, c_3_3 = 1, -2/3, 2/3
d_3_1, d_3_2, d_3_3 = -1/24, 3/4, 7/24
# Fourth Order
c_4_1 = c_4_4 = 1/(2*(2-2**(1/3)))
c_4_2 = c_4_3 = (1-2**(1/3))/(2*(2-2**(1/3)))
d_4_1 = d_4_3 = 1/(2-2**(1/3))
d_4_2, d_4_4 = -2**(1/3)/(2-2**(1/3)), 0


def odeint(f: callable, ic: Tuple, t: ndarray, method: str = 'rk4', terminate: callable = lambda *args: False,
           fargs: Tuple = ()) -> ndarray:

    if len(ic) == 2:
        symplectic = True
    elif len(ic) == 1:
        symplectic = False
    else:
        print("'ic' Expects Only One Or Two Inputs, Aborting...")
        return None

    if method == 'euler':
        if symplectic:
            print("'ic' expected only one input for Euler's Method")
            return None
        else:
            step = euler_step
    elif method == 'heun':
        if symplectic:
            print("'ic' expected only one input for Heun's Method")
            return None
        else:
            step = heun_step
    elif method == 'rk2a':
        if symplectic:
            print("'ic' expected only one input for the RK2a Method")
            return None
        else:
            step = rk2a_step
    elif method == 'rk2b':
        if symplectic:
            print("'ic' expected only one input for the RK2b Method")
            return None
        else:
            step = rk2b_step
    elif method == 'rk4':
        if symplectic:
            print("'ic' expected only one input for the RK4 Method")
            return None
        else:
            step = rk4_step
    elif method == 'eulercromer':
        if not symplectic:
            print("'ic' expected two inputs for The Euler-Cromer Method")
            return None
        else:
            step = eulercromer_step
    elif method == 'verlet':
        if not symplectic:
            print("'ic' expected 2 inputs for Verlet's Method")
            return None
        else:
            step = verlet_step
    elif method == 'symplectic_3rd_order':
        if not symplectic:
            print("'ic' expected 2 inputs for The 3rd Order Simplectic Integrator Method")
            return None
        else:
            step = symplectic_3rd_order_step
    elif method == 'symplectic_4th_order':
        if not symplectic:
            print("'ic' expected 2 inputs for The 4th Order Simplectic Integrator Method")
            return None
        else:
            step = symplectic_4th_order_step
    else:
        if len(ic)==1:
            print("Method Not Recognized, Defaulting To RK4")
            step = rk4
        elif len(ic)==2:
            print("Method Not Recognized, Defaulting To Euler-Cromer")
            step = eulercromer
        else:
            print("Method Not Recognized, Failed To Detect Default Appropriate Method")
            return None

    n = np.size(t)
    x = np.zeros((n,np.size(ic[0])))
    x[0] = np.asarray(ic[0])

    # Simplectic case gets its own block so as to not have to redundantly check
    # whether its symplectic or not in the loop
    if symplectic:
        dx = np.zeros((n,np.size(ic[1])))
        dx[0] = np.asarray(ic[1])

        for i in range(n - 1):
            x[i + 1], dx[i+1] = step(f, t[i], t[i + 1] - t[i], x[i], dx[i])

            for func in fargs:
                x[i+1], dx[i+1] = func(x[i+1], dx[i+1])

            if terminate(x[i+1]):
                dx = dx[:i + 1]
                x = x[:i+1]
                t = t[:i+1]
                break
        return x.T, dx.T
    else:
        for i in range(n - 1):
            x[i + 1] = step(f, t[i], t[i + 1] - t[i], x[i])

            for func in fargs:
                x[i + 1] = func(x[i + 1])

            if terminate(x[i + 1]):
                x = x[:i + 1]
                t = t[:i + 1]
                break
        return x.T


def euler(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
          terminate: callable = lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1] = func(x[i+1])

        if terminate(x[i+1]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def heun(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1] = func(x[i+1])

        if terminate(x[i+1]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk2a(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1] = func(x[i+1])

        if terminate(x[i+1]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk2b(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
         terminate: callable = lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1] = func(x[i+1])

        if terminate(x[i+1]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def rk4(f: callable, x0: Union[ndarray, List, float], t: Union[ndarray, List],
        terminate: callable = lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1] = func(x[i+1])

        if terminate(x[i+1]):
            x = x[:i+1]
            t = t[:i+1]
            break
    return x.T


def eulercromer(f, x0, dx0, t, terminate=lambda *args: False, fargs: Tuple = ()) -> ndarray:
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

        for func in fargs:
            x[i+1], dx[i+1] = func(x[i+1], dx[i+1])

        if terminate(x[i+1], dx[i+1]):
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

        if x[i+1,0]<-np.pi:
            x[i+1,0] += 2*np.pi
        elif x[i+1, 0] > np.pi:
            x[i+1, 0] -= 2 * np.pi
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

        if x[i+1,0]<-np.pi:
            x[i+1,0] += 2*np.pi
        elif x[i+1, 0] > np.pi:
            x[i+1, 0] -= 2 * np.pi
    return x.T


def verlet(f, x0, dx0, t, terminate=lambda *args: False, fargs: Tuple = ()) -> ndarray:
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
        dx1 = dx[i] + d_2_1 * (t[i + 1] - t[i]) * f(t[i], x[i], dx[i])
        x1 = x[i] + c_2_1 * (t[i + 1] - t[i]) * dx1
        dx[i+1] = dx1 + d_2_2 * (t[i + 1] - t[i]) * f(t[i], x1, dx1)
        x[i+1] = x1 + c_2_2 * (t[i + 1] - t[i]) * dx[i+1]

        for func in fargs:
            x[i+1], dx[i+1] = func(x[i+1], dx[i+1])

        if terminate(x[i+1], dx[i+1]):
            dx = dx[:i + 1]
            x = x[:i+1]
            t = t[:i+1]
            break

    return x.T, dx.T


def symplectic_3rd_order(f, x0, dx0, t, terminate=lambda *args: False, fargs: Tuple = ()) -> ndarray:
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
        dx1 = dx[i] + d_3_1 * (t[i + 1] - t[i]) * f(t[i], x[i], dx[i])
        x1 = x[i] + c_3_1 * (t[i + 1] - t[i]) * dx1
        dx2 = dx1 + d_3_2 * (t[i + 1] - t[i]) * f(t[i], x1, dx1)
        x2 = x1 + c_3_2 * (t[i + 1] - t[i]) * dx2
        dx[i+1] = dx2 + d_3_3 * (t[i + 1] - t[i]) * f(t[i], x2, dx2)
        x[i+1] = x2 + c_3_3 * (t[i + 1] - t[i]) * dx[i+1]

        for func in fargs:
            x[i+1], dx[i+1] = func(x[i+1], dx[i+1])

        if terminate(x[i+1], dx[i+1]):
            dx = dx[:i + 1]
            x = x[:i+1]
            t = t[:i+1]
            break

    return x.T, dx.T


def symplectic_4th_order(f, x0, dx0, t, terminate=lambda *args: False, fargs: Tuple = ()) -> ndarray:
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
        dx1 = dx[i] + d_4_1 * (t[i + 1] - t[i]) * f(t[i], x[i], dx[i])
        x1 = x[i] + c_4_1 * (t[i + 1] - t[i]) * dx1
        dx2 = dx1 + d_4_2 * (t[i + 1] - t[i]) * f(t[i], x1, dx1)
        x2 = x1 + c_4_2 * (t[i + 1] - t[i]) * dx2
        dx3 = dx2 + d_4_3 * (t[i + 1] - t[i]) * f(t[i], x2, dx2)
        x3 = x2 + c_4_3 * (t[i + 1] - t[i]) * dx3
        dX = dx3 + d_4_4 * (t[i + 1] - t[i]) * f(t[i], x3, dx3)
        x[i+1] = x3 + c_4_4 * (t[i + 1] - t[i]) * dx[i+1]

        for func in fargs:
            x[i+1], dx[i+1] = func(x[i+1], dx[i+1])

        if terminate(x[i+1], dx[i+1]):
            dx = dx[:i + 1]
            x = x[:i+1]
            t = t[:i+1]
            break

    return x.T, dx.T

####################################

def euler_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float]) -> ndarray:
    return x + dt*f(t, x)


def heun_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float]) -> ndarray:
    k1 = dt * f(t, x)
    k2 = dt * f(t+dt, x + k1)
    return x + (k1 + k2) / 2


def rk2a_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float]) -> ndarray:
    k1 = 0.5*dt * f(t, x)
    return x + dt * f(t + 0.5*dt, x + k1)


def rk2b_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float]) -> ndarray:
    k1 = dt * f(t, x)
    k2 = dt * f(t+dt, x + k1)
    return x + 0.5*(k1 + k2)


def rk4_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float]) -> ndarray:
    k1 = dt * f(t, x)
    k2 = dt * f(t + 0.5 * dt, x + 0.5 * k1)
    k3 = dt * f(t + 0.5 * dt, x + 0.5 * k2)
    k4 = dt * f(t + dt, x + k3)
    return x + (k1 + 2 * (k2 + k3) + k4) / 6


def eulercromer_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float], dx: Union[ndarray, List, float]) -> ndarray:
    dx = dx + dt * f(t, x, dx)
    return x + dt * dx, dx


def verlet_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float], dx: Union[ndarray, List, float]) -> ndarray:
    dx1 = dx + d_2_1 * dt * f(t, x, dx)
    x1 = x + c_2_1 * dt * dx1
    dx = dx1 + d_2_2 * dt * f(t, x1, dx1)
    return x1 + c_2_2 * dt * dx, dx


def symplectic_3rd_order_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float], dx: Union[ndarray, List, float]) -> ndarray:
    dx1 = dx + d_3_1 * dt * f(t, x, dx)
    x1 = x + c_3_1 * dt * dx1
    dx2 = dx1 + d_3_2 * dt * f(t, x1, dx1)
    x2 = x1 + c_3_2 * dt * dx2
    dx = dx2 + d_3_3 * dt * f(t, x2, dx2)
    return x2 + c_3_3 * dt * dx, dx


# TODO: Fix `symplectic_4th_order_step`


def symplectic_4th_order_step(f: callable, t: float, dt: float, x: Union[ndarray, List, float], dx: Union[ndarray, List, float]) -> ndarray:
    dx1 = dx + d_4_1 * dt * f(t, x, dx)
    x1 = x + c_4_1 * dt * dx1
    dx2 = dx1 + d_4_2 * dt * f(t, x1, dx1)
    x2 = x1 + c_4_2 * dt * dx2
    dx3 = dx2 + d_4_3 * dt * f(t, x2, dx2)
    x3 = x2 + c_4_3 * dt * dx3
    dx = dx3 + d_4_4 * dt * f(t, x3, dx3)
    return x3 + c_4_4 * dt * dx, dx
