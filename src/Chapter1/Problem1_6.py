########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 6                                                                            #
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


def rhs(t, X, a, b):
    return np.array([a*X[0]-b*X[0]**2, ])


N0_1, N0_2, N0_3 = 1, 1, 1
a1, a2, a3 = 1, 1, 1
b1, b2, b3 = 0, 3, 0.01
t = np.linspace(0,20,200)  # dt = 0.1
y1 = euler(partial(rhs, a=a1, b=b1), [N0_1, ], t)
y2 = euler(partial(rhs, a=a2, b=b2), [N0_2, ], t)
y3 = euler(partial(rhs, a=a3, b=b3), [N0_3, ], t)

# Plotting
fig, axs = plt.subplots(3, 1, sharex=True)

axs[0].plot(t, y1[0])
axs[1].plot(t, y2[0])
axs[2].plot(t, y3[0])

for ax in axs:
    ax.grid()
    ax.set_xlabel("t")
    ax.set_ylabel("N")

axs[0].set_title(rf"$\frac{{dN}}{{dt}}={a1}N-{b1}N^2$")
axs[1].set_title(rf"$\frac{{dN}}{{dt}}={a2}N-{b2}N^2$")
axs[2].set_title(rf"$\frac{{dN}}{{dt}}={a3}N-{b3}N^2$")

plt.suptitle("Problem 1.6")
plt.subplots_adjust(hspace=0.45)
plt.savefig("../../figures/Chapter1/Problem1_6", dpi=300)
