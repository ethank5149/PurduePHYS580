########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 10                                                                           #
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
from lib.DSolve import eulercromer, eulercromer_pendulum
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
l = g = 9.81
q = 0.5
Omega_D = 2/3


def f(t, X, dX, F_D):
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0]+F_D*np.sin(Omega_D*t),])


t = np.linspace(0,60,1400)  # dt = 0.04

x1, dx1 = eulercromer_pendulum(partial(f,F_D=0.1),[0.2,],[0],t)
x2, dx2 = eulercromer_pendulum(partial(f,F_D=0.5),[0.2,],[0],t)
x3, dx3 = eulercromer_pendulum(partial(f,F_D=0.99),[0.2,],[0],t)

fig, ax = plt.subplots(1, 2)

ax[0].plot(t, x1[0], label=r"$F_D=0.1$")
ax[0].plot(t, x2[0], label=r"$F_D=0.5$")
ax[0].plot(t, x3[0], label=r"$F_D=0.99$")
ax[0].grid()
ax[0].set_xlabel(r"$t\,[s]$")
ax[0].set_ylabel(r"$\theta\,[rad]$")
ax[0].legend()
ax[0].set_title("Waveforms")

ax[1].plot(x1[0],dx1[0], label=r"$F_D=0.1$")
ax[1].plot(x2[0],dx2[0], label=r"$F_D=0.5$")
ax[1].plot(x3[0],dx3[0], label=r"$F_D=0.99$")
ax[1].grid()
ax[1].set_xlabel(r"$\theta [rad]$")
ax[1].set_ylabel(r"$\dot{\theta} [rad/s]$")
ax[1].legend()
ax[1].set_title("Phase Plots")

plt.suptitle("Problem 3.10")
plt.savefig("../../figures/Chapter3/Problem3_10", dpi=300)

