import numpy as np
import matplotlib.pyplot as plt

hbar = 1
m = 1
k = 1
omega = np.sqrt(k / m)
L = 1
V0 = 50
Vinf = 100000
dpsi_0 = 1
zero = 1e-11
eps = 1e-8


def e_ipw(n):
    return (np.pi * hbar * n / L) ** 2 / (2 * m)


def e_qho(n):
    return hbar * omega * (n + 0.5)


def V_wall(x):
    if abs(x) > L:
        return Vinf
    elif abs(x) < L / 50:
        return V0
    else:
        return 0


def V_double_wall(x):
    if -L/10 < x < -2*L/10:
        return V0
    if 3*L/10 < x < 4*L/10:
        return V0
    else:
        return 0


def V_ipw(x):
    return 0 if abs(x) < L else Vinf


def V_qho(x):
    return 0.5 * m * omega ** 2 * x ** 2


class QM:
    def __init__(self, psi0, domain, bval=0, V=lambda *args: 0):
        self.domain = domain
        self.psi = np.zeros((2, np.size(self.domain)))
        self.psi[:, 0] = psi0
        self.bval = bval
        self.V = V

    def TISE(self, psi, x, E):
        return np.asarray([psi[1], (2 * m / hbar ** 2) * (self.V(x) - E) * psi[0]])

    def norm(self):
        # trapezoidal rule
        sum = 0
        for i in range(np.size(self.domain) - 1):
            dx = self.domain[i + 1] - self.domain[i]
            f_i = (np.real(self.psi[0, i]) + np.imag(self.psi[0, i])) * (
                    np.real(self.psi[0, i]) - np.imag(self.psi[0, i]))
            f_ip1 = (np.real(self.psi[0, i + 1]) + np.imag(self.psi[0, i + 1])) * (
                    np.real(self.psi[0, i + 1]) - np.imag(self.psi[0, i + 1]))
            sum = sum + 0.5 * (f_i + f_ip1) * dx
        self.psi[0, :] = self.psi[0, :] / np.sqrt(sum)

    def normsqr(self):
        return (np.real(self.psi) + np.imag(self.psi)) * (np.real(self.psi) - np.imag(self.psi))

    def solve(self, alpha):
        n = np.size(self.domain)
        # rk4
        for i in range(n - 1):
            h = self.domain[i + 1] - self.domain[i]
            k1 = h * self.TISE(self.psi[:, i], self.domain[i], alpha)
            k2 = h * self.TISE(self.psi[:, i] + 0.5 * k1, self.domain[i] + 0.5 * h, alpha)
            k3 = h * self.TISE(self.psi[:, i] + 0.5 * k2, self.domain[i] + 0.5 * h, alpha)
            k4 = h * self.TISE(self.psi[:, i] + k3, self.domain[i + 1], alpha)
            self.psi[:, i + 1] = self.psi[:, i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        self.norm()

    def obj_func(self, alpha):
        self.solve(alpha)
        return self.psi[0, -1] - self.bval

    def shoot(self, a, b, tol=eps):
        # regula falsi
        c = 0
        while abs(b - a) > tol:
            fa, fb = self.obj_func(a), self.obj_func(b)
            c = (a * fb - b * fa) / (fb - fa)
            fc = self.obj_func(c)
            print(a, b, c)
            print(fa, fb, fc)

            if abs(fc) < zero:
                return c
            elif fa * fc > 0:
                a = c
            else:
                b = c
        return c
        # if self.obj_func(a) * self.obj_func(b) < 0.0:
        #     # bisection
        #     c = 0
        #     while abs(b - a) > tol:
        #         if max(abs(a),abs(b))>max(abs(a0),abs(b0)):
        #             print("Uh, oh")
        #         c = 0.5 * (a + b)
        #         fa, fb, fc = self.obj_func(a), self.obj_func(b), self.obj_func(c)
        #
        #         if abs(fa) < zero:
        #             return a
        #         if abs(fb) < zero:
        #             return b
        #         if abs(fc) < zero:
        #             return c
        #
        #         if fa * fc > 0:
        #             a = c
        #         elif fb * fc > 0:
        #             b = c
        #         else:
        #             print("Uh, oh. Something's fishy\n")
        #     return c
        # else:
        #     # regula falsi
        #     c = 0
        #     while abs(b - a) > tol:
        #         fa, fb = self.obj_func(a), self.obj_func(b)
        #         c = (a * fb - b * fa) / (fb - fa)
        #         fc = self.obj_func(c)
        #         print(a, b, c)
        #         print(fa, fb, fc)
        #
        #         if abs(fc) < zero:
        #             return c
        #         elif fa * fb < 0:
        #             b = c
        #         else:
        #             a = c
        #     return c


def main():
    L = 1
    domain = np.linspace(-L / 2, L / 2, 1000)
    psi0 = np.array([0, dpsi_0])
    qm = QM(psi0, domain, V=V_double_wall)

    E = qm.shoot(0, 5)
    print(E)
    qm.solve(E)
    plt.plot(domain, E + qm.normsqr()[0])
    plt.fill_between(domain, [qm.V(x) for x in domain], label="V(x)", color='black', alpha=0.5)
    plt.grid()
    plt.legend()
    plt.show()


main()
# 5,20,45,80