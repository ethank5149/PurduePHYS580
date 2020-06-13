########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 1                                                                            #
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
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
P = 400  # Power [W]
m = 70  # mass [kg]
v0 = 4  # Initial velocity [m/s]


def rhs(t, X):
    return np.array([P/(m*X[0]), ])


def v(t):
    return np.sqrt(2*P*t/m+v0**2)


t1 = np.linspace(0, 200, 400)  # dt = 0.5
t2 = np.linspace(0, 200, 800)  # dt = 0.25
t3 = np.linspace(0, 200, 1600)  # dt = 0.125

y1 = euler(rhs, [v0, ], t1)
y2 = euler(rhs, [v0, ], t2)
y3 = euler(rhs, [v0, ], t3)


# Plotting
fig, ax = plt.subplots(2, 1, sharex=True)

ax[0].plot(t1, y1[0], label="dt = 0.5")
ax[0].plot(t2, y2[0], label="dt = 0.25")
ax[0].plot(t3, y3[0], label="dt = 0.125")
ax[1].plot(t1, y1[0] - v(t1), label="dt = 0.5")
ax[1].plot(t2, y2[0] - v(t2), label="dt = 0.25")
ax[1].plot(t3, y3[0] - v(t3), label="dt = 0.125")

ax[0].legend()
ax[0].grid()
ax[0].set_xlabel("t")
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel("t")

ax[0].set_ylabel(r"$v(t)$")
ax[1].set_ylabel(r"Error")

plt.suptitle("Problem 2.1")
plt.savefig("../../figures/Chapter2/Problem2_1", dpi=300)
