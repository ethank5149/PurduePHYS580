########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 2                                                                            #
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
from lib.NDSolveSystem import ODE, SymplecticODE
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X, m):
    return np.array([-m*g, ])


def exact(t, m, dx_0):
    return dx_0-m*g*t


m = 1
dx_0 = 40

sim = ODE(partial(rhs, m=m), np.array([dx_0, ]), ti=0, dt=0.1, tf=10)
sim.run()

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

ax1.plot(sim.t, sim.X_series[0], label="Model")
ax1.plot(sim.t, exact(sim.t, m, dx_0), label="Exact")
ax1.legend()
ax1.grid()
ax1.set_xlabel("t")
ax1.set_ylabel(r"$\frac{dx}{dt}$")

ax2.plot(sim.t, sim.X_series[0]-exact(sim.t, m, dx_0))
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel("Error")

ax3.plot(sim.t, 100*(sim.X_series[0] -
                     exact(sim.t, m, dx_0))/exact(sim.t, m, dx_0))
ax3.grid()
ax3.set_xlabel("t")
ax3.set_ylabel("% Error")

plt.suptitle("Problem 1.2")
plt.subplots_adjust(hspace=0.45)
plt.savefig("../../figures/Chapter1/Problem1_2", dpi=300)
