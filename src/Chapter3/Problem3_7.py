########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 7                                                                            #
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
l = 1
Omega = np.sqrt(g/l)

def f(t, X, dX, q, F_D, Omega_D):
    return np.array([-(g/l)*X[0]-q*dX[0]+F_D*np.sin(Omega_D*t),])


def omega(q):
    return np.sqrt(Omega**2-q**2/4)


t = np.linspace(0,20,2000)
x1, dx1 = eulercromer(partial(f,F_D=1, Omega_D=0.5*omega(2), q=2),[1,],[0],t)
x2, dx2 = eulercromer(partial(f,F_D=1, Omega_D=1*omega(2), q=2),[1,],[0],t)
x3, dx3 = eulercromer(partial(f,F_D=1, Omega_D=1.5*omega(2), q=2),[1,],[0],t)
x4, dx4 = eulercromer(partial(f,F_D=1, Omega_D=0.5*omega(1), q=1),[1,],[0],t)
x5, dx5 = eulercromer(partial(f,F_D=1, Omega_D=1*omega(1), q=1),[1,],[0],t)
x6, dx6 = eulercromer(partial(f,F_D=1, Omega_D=1.5*omega(1), q=1),[1,],[0],t)
x7, dx7 = eulercromer(partial(f,F_D=1, Omega_D=0.5*omega(0), q=0),[1,],[0],t)
x8, dx8 = eulercromer(partial(f,F_D=1, Omega_D=1*omega(0), q=0),[1,],[0],t)
x9, dx9 = eulercromer(partial(f,F_D=1, Omega_D=1.5*omega(0), q=0),[1,],[0],t)

fig, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].set_title(r"$q=2$")
ax[0].plot(t, x1[0], label=r"$\Omega_D=0.5\Omega$")
ax[0].plot(t, x2[0], label=r"$\Omega_D=\Omega$")
ax[0].plot(t, x3[0], label=r"$\Omega_D=1.5\Omega$")
ax[0].grid()
ax[0].set_xlabel("t [s]")
ax[0].set_ylabel("x [m]")
ax[0].legend()

ax[1].set_title(r"$q=1$")
ax[1].plot(t, x4[0], label=r"$\Omega_D=0.5\Omega$")
ax[1].plot(t, x5[0], label=r"$\Omega_D=\Omega$")
ax[1].plot(t, x6[0], label=r"$\Omega_D=1.5\Omega$")
ax[1].grid()
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel("x [m]")
ax[1].legend()

ax[2].set_title(r"$q=0$")
ax[2].plot(t, x7[0], label=r"$\Omega_D=0.5\Omega$")
ax[2].plot(t, x8[0], label=r"$\Omega_D=\Omega$")
ax[2].plot(t, x9[0], label=r"$\Omega_D=1.5\Omega$")
ax[2].grid()
ax[2].set_xlabel("t [s]")
ax[2].set_ylabel("x [m]")
ax[2].legend()

plt.suptitle("Problem 3.7")
plt.savefig("../../figures/Chapter3/Problem3_7", dpi=300)

