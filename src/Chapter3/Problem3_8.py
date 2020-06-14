########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 8                                                                            #
#      \\          |                                                                                                   #
#      //          |  Author: Ethan Knox                                                                               #
#     //           |  Website: https://www.github.com/ethank5149                                                       #
#     ========     |  MIT License                                                                                      #
########################################################################################################################
########################################################################################################################
# License                                                                                                              #
# Copyright 2020 Ethan Knox                                                                                            #
#                                                                                                                      #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated         #
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the  #
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to      #
# permit persons to whom the Software is furnished to do so, subject to the following conditions:                      #
#                                                                                                                      #
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the #
# Software.                                                                                                            #
#                                                                                                                      #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE #
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS  #
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR  #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.     #
########################################################################################################################

# Including
from lib.DSolve import rk4, eulercromer
from lib.FindRoot import secant
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
m = 1
g = 9.81
q = 1
F_D = 1



def f(t, X, dX, Omega_D, l):
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0]+F_D*np.sin(Omega_D*t),])


def omega(l):
    return np.sqrt(Omega(l)**2-q**2/4)


def T(l):
    return 2*np.pi*np.sqrt(l/g)


def Omega(l):
    return np.sqrt(g/l)

t = np.linspace(0,20,2000)
x1, dx1 = eulercromer(partial(f, Omega_D=0.75*omega(1), l=1),[np.pi/2.5,],[0],t)
x2, dx2 = eulercromer(partial(f, Omega_D=0.75*omega(2), l=2),[np.pi/2.5,],[0],t)
x3, dx3 = eulercromer(partial(f, Omega_D=0.75*omega(4), l=4),[np.pi/2.5,],[0],t)
x4, dx4 = eulercromer(partial(f, Omega_D=0.75*omega(8), l=8),[np.pi/2.5,],[0],t)
x5, dx5 = eulercromer(partial(f, Omega_D=0.75*omega(16), l=16),[np.pi/2.5,],[0],t)

fig, ax = plt.subplots(1, 1)
ax.plot(t, x1[0], label=rf"$T={T(1):0.4f}$")
ax.plot(t, x2[0], label=rf"$T={T(2):0.4f}$")
ax.plot(t, x3[0], label=rf"$T={T(4):0.4f}$")
ax.plot(t, x4[0], label=rf"$T={T(8):0.4f}$")
ax.plot(t, x5[0], label=rf"$T={T(16):0.4f}$")
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [m]")
ax.legend()
ax.set_title(r"$q=1,\,\,\Omega_D=0.75\Omega$")

plt.suptitle("Problem 3.8")
plt.savefig("../../figures/Chapter3/Problem3_8", dpi=300)

