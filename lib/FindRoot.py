# Global Definitions
EPS = 2.22044604925e-16
ZERO = 1e-14
MAX_ITERATIONS = 10000


class MaxIterWarning(Warning):
    pass


class UserInputError(BaseException):
    pass


def bisection(f: callable, a: float, c: float, tol: float = None) -> float:
    r"""Finds a root using the bisection method.

    Parameters
    ----------
    f : callable
        Any callable, univariate function.
    a : float
        Starting point of the bracket used to surround the root.
    c : float
        Endpoint of the bracket used to surround the root.
    tol : Tolerance

    Returns
    -------
    answer : float or None
        Will either return the root, or will raise a `UserInputError` and
        return None.

    """

    if tol is None:
        tolerance = 0.5 * EPS * (abs(a) + abs(c))
    else:
        tolerance = tol

    f_a = f(a)
    f_c = f(c)

    # Check that sign(f(a)) != sign(f(b)).
    # If not, the algorithm can't keep track of where the root is to enclose around it.
    if f_a*f_c > 0.0:
        raise UserInputError("Bracket must enclose an odd number of roots.")

    for iteration in range(MAX_ITERATIONS):
        b = 0.5 * (a + c)  # Bisect
        f_b = f(b)  # Function value at midpoint

        # If f(b) = 0 or the bracket is sufficiently small, stop and return b, the current midpoint
        if (abs(c - a) < tolerance) or (abs(f_b) < ZERO):
            return b

        # If function changes sign between x = a and x = b
        if f_b*f_a <= 0.0:
            # Replace rightmost bound(c) with the new midpoint(b)
            c = b
        else:
            # Otherwise replace leftmost bound(a) with the new midpoint(b)
            a = b

        # Update the bracket function values
        f_a = f(a)
        f_c = f(c)
    return b


def secant(f, x1, x2):
    r"""Finds a root of a given univariate function using the secant method.

    Parameters
    ----------
    f : callable
        Any callable, univariate function.
    x1 : float
        Starting point of the bracket used to surround the root.
    x2 : float
        Endpoint of the bracket used to surround the root.

    Returns
    -------
    answer : float or None
        Will either return the root, or will raise a `UserInputError` and
        return None.

    """
    tolerance = 0.5 * EPS * (abs(x1) + abs(x2))

    f_x1 = f(x1)
    f_x2 = f(x2)

    if abs(f_x1) < abs(f_x2):
        b = x1
        a = x2
        f_b = f(b)
        f_a = f(a)
    else:
        a = x1
        b = x2
        f_a = f(a)
        f_b = f(b)

    for j in range(MAX_ITERATIONS):
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)

        if (abs(c - b) < tolerance) or (abs(f_c) < ZERO):
            return c

        a = b
        b = c
        f_a = f_b
        f_b = f_c

    return c
