########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 5                                                                            #
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
from lib.Constants import Air
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


P = 400  # Power [W]
m = 70  # Mass [kg]
v0 = 4  # Initial velocity [m/s]
C = 0.5  # Drag coefficient
v_star = 7
F_initial = P/v_star
A = 0.33

air = Air()
rho = air.density


def rhs(t, X, F0):
    if F0*X[0] < P:
        return np.array([F0/m - 0.5 * C * rho * A * X[0] ** 2 / m, ])
    else:
        return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m, ])


def rhs_old(t, X):
    return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m, ])


curried_rhs = partial(rhs, F0=F_initial)
t = np.linspace(0,50,5000)  # dt = 0.01

y1 = euler(curried_rhs, [v0, ], t)
y2 = euler(rhs_old, [v0, ], t)

# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(t, y1[0], label=f"New Method")
ax.plot(t, y2[0], label=f"Old Method")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")


plt.suptitle("Problem 2.5")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_5", dpi=300)
