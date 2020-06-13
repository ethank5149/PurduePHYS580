########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 1 - Problem 5                                                                            #
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


def rhs(t, X, ta, tb):
    return np.array([X[1]/tb-X[0]/ta, X[0]/ta-X[1]/tb])


def Na(t, a, b, Na0, Nb0):
    return Na0*(b*np.exp(-(a+b)*t/(a*b))+a)/(a+b)-Nb0*a*(np.exp(-(a+b)*t/(a*b))-1)/(a+b)


def Nb(t, a, b, Na0, Nb0):
    return Nb0*(a*np.exp(-(a+b)*t/(a*b))+b)/(a+b)-Na0*b*(np.exp(-(a+b)*t/(a*b))-1)/(a+b)


ta1, ta2, ta3 = 1, 2, 4
tb1, tb2, tb3 = 4, 2, 1
Na0, Nb0 = 1, 0
t = np.linspace(0,20,200)  # dt = 0.1
y1 = euler(partial(rhs, ta=ta1, tb=tb1),[Na0, Nb0],t)
y2 = euler(partial(rhs, ta=ta2, tb=tb2),[Na0, Nb0],t)
y3 = euler(partial(rhs, ta=ta3, tb=tb3),[Na0, Nb0],t)

# Plotting
fig, axs = plt.subplots(2, 3)

axs[0, 0].plot(t, y1[0], label=r"$N_A$")
axs[0, 0].plot(t, y1[1], label=r"$N_B$")
axs[0, 1].plot(t, y2[0], label=r"$N_A$")
axs[0, 1].plot(t, y2[1], label=r"$N_B$")
axs[0, 2].plot(t, y3[0], label=r"$N_A$")
axs[0, 2].plot(t, y3[1], label=r"$N_B$")
axs[1, 0].plot(t, y1[0] - Na(t, ta1, tb1, Na0, Nb0), label=r"$N_A$")
axs[1, 0].plot(t, y1[1] - Nb(t, ta1, tb1, Na0, Nb0), label=r"$N_B$")
axs[1, 1].plot(t, y2[0] - Na(t, ta2, tb2, Na0, Nb0), label=r"$N_A$")
axs[1, 1].plot(t, y2[1] - Nb(t, ta2, tb2, Na0, Nb0), label=r"$N_B$")
axs[1, 2].plot(t, y3[0] - Na(t, ta3, tb3, Na0, Nb0), label=r"$N_A$")
axs[1, 2].plot(t, y3[1] - Nb(t, ta3, tb3, Na0, Nb0), label=r"$N_B$")

for i in range(2):
    for j in range(3):
        axs[i, j].legend()
        axs[i, j].grid()
        axs[i, j].set_xlabel("t")
        axs[0, j].set_ylabel(r"$N$")
        axs[1, j].set_ylabel(r"Error")

axs[0, 0].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta1}}}{{{tb1}}}$")
axs[0, 1].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta2}}}{{{tb2}}}$")
axs[0, 2].set_title(rf"$\frac{{\tau_A}}{{\tau_B}} = \frac{{{ta3}}}{{{tb3}}}$")
plt.suptitle("Problem 1.5")
for ax in axs.flat:
    ax.label_outer()
plt.subplots_adjust(hspace=0.2, wspace=0.3)

plt.savefig("../../figures/Chapter1/Problem1_5", dpi=300)
