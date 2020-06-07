###############################################################################
#     ========     |  Purdue Physics 580 - Computational Physics              #
#     \\           |  Chapter 1 - Exercise 2                                  #
#      \\          |                                                          #
#      //          |  Author: Ethan Knox                                      #
#     //           |  Website: https://www.github.com/ethank5149              #
#     ========     |  MIT License                                             #
###############################################################################
###############################################################################
# License                                                                     #
# Copyright 2020 Ethan Knox                                                   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
###############################################################################

# Including
###############################################################################
from NDSolveSystem import ODE
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
###############################################################################

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X, m):
    return np.array([X[1], -m*g])


def rhs_exact(t, m, x0_0, x1_0):
    return np.array([x0_0+x1_0*t-0.5*m*g*t**2, x1_0-m*g*t])


def terminate(X):
    return X[0] < 0


def main():
    m = 1
    x0_0 = 100
    x1_0 = -10

    sim = ODE(partial(rhs, m=m), np.array(
        [x0_0, x1_0]), ti=0, dt=0.0001, tf=10, terminate=terminate)
    sim.run()
    sim.store()

    # Plotting
    ###########################################################################
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(sim.t_series, sim.X_series[:, 0], label=r"y")
    ax.plot(sim.t_series, sim.X_series[:, 1], label=r"$\dot{y}$")
    ax.legend()
    ax.grid()
    ax.set_xlabel("t")
    plt.show()


main()
