########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 2                                                                            #
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
from lib.DSolve import euler
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
P = 400  # Power [W]
m = 70  # Mass [kg]
v0 = 4  # Initial velocity [m/s]
C = 0.5  # Drag coefficient
A = 0.33  # Cross-sectional area [m^2]
air = Constants.Air()
rho = air.density  # Air density [kg/m^3]


def rhs(t, X, A):
    return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m, ])


def rhs2(t, X, A):
    return np.array([X[1], P/(m*X[1])-0.5*C*rho*A*X[1]**2/m])


def W_d(X, A):
    return X[0]*0.5*C*rho*A*X[1]**2


def v(t):
    return np.sqrt(2*P*t/m+v0**2)


t = np.linspace(0,200,20000)
ic = np.array([v0, ])
ic2 = np.array([0, v0])

y1 = euler(partial(rhs, A=A), ic, t)
y2 = euler(partial(rhs, A=0.3*A), ic, t)
y3 = euler(partial(rhs2, A=A), ic2, t)
y4 = euler(partial(rhs2, A=0.3*A), ic2, t)

# Plotting
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(t, y1[0], label="Front of the pack")
ax[0].plot(t, y2[0], label="Middle of the pack")
ax[1].plot(t, W_d(y3, A)-W_d(y4, 0.3*A), label=r"$W_d$ Front - $W_d$ Middle")

ax[0].legend()
ax[0].grid()
ax[0].set_xlabel("t")
ax[0].set_ylabel(r"$v(t)$")

ax[1].legend()
ax[1].grid()
ax[1].set_xlabel("t")
ax[1].set_ylabel(r"$W [J]$")

plt.suptitle("Problem 2.2")
plt.savefig("../../figures/Chapter2/Problem2_2", dpi=300)
