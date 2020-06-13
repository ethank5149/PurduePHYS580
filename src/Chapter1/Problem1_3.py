########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 3                                                                            #
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
import numpy as np
from lib.DSolve import euler
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X, a, b):
    return np.array([a-b*X[0], ])


a = 9.81
dx_0 = 0
b1, b2, b3 = 1, 2, 3

t = np.linspace(0,10,100)  # dt = 0.1
y1 = euler(partial(rhs, a=a, b=b1), [dx_0, ], t)[0]
y2 = euler(partial(rhs, a=a, b=b2), [dx_0, ], t)[0]
y3 = euler(partial(rhs, a=a, b=b3), [dx_0, ], t)[0]

# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(t, y1, label=f"b = {b1}")
ax.plot(t, y2, label=f"b = {b2}")
ax.plot(t, y3, label=f"b = {b3}")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")

plt.suptitle("Problem 1.3")
plt.subplots_adjust(hspace=0.45)
plt.savefig("../../figures/Chapter1/Problem1_3", dpi=300)
