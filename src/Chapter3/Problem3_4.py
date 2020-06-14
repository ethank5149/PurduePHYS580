########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 4                                                                            #
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
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions


def f(t, X, dX, k, alpha):
    return np.array([-k*X[0]**alpha,])


t = np.linspace(0,20,2000)
x1, dx1 = eulercromer(partial(f,alpha=1, k=1),[1,],[0],t)
x2, dx2 = eulercromer(partial(f,alpha=1, k=1),[2,],[0],t)
x3, dx3 = eulercromer(partial(f,alpha=1, k=1),[3,],[0],t)
x4, dx4 = eulercromer(partial(f,alpha=1, k=1),[4,],[0],t)

fig, ax = plt.subplots(1, 1)
ax.plot(t, x1[0], label=r"$A_0 = 1$")
ax.plot(t, x2[0], label=r"$A_0 = 2$")
ax.plot(t, x3[0], label=r"$A_0 = 3$")
ax.plot(t, x4[0], label=r"$A_0 = 4$")
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel("Amplitude")
ax.set_title("Varying Initial Amplitudes - Harmonic Oscillator")
plt.legend()
plt.suptitle("Problem 3.4a")
plt.savefig("../../figures/Chapter3/Problem3_4a", dpi=300)


x1, dx1 = eulercromer(partial(f,alpha=3, k=1),[1,],[0],t)
x2, dx2 = eulercromer(partial(f,alpha=3, k=1),[0.8,],[0],t)
x3, dx3 = eulercromer(partial(f,alpha=3, k=1),[0.4,],[0],t)
x4, dx4 = eulercromer(partial(f,alpha=3, k=1),[0.2,],[0],t)

fig, ax = plt.subplots(1, 1)
ax.plot(t, x1[0], label=r"$A_0 = 1$")
ax.plot(t, x2[0], label=r"$A_0 = 0.8$")
ax.plot(t, x3[0], label=r"$A_0 = 0.4$")
ax.plot(t, x4[0], label=r"$A_0 = 0.2$")
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel("Amplitude")
ax.set_title("Varying Initial Amplitudes - Anharmonic Oscillator")
plt.legend()
plt.suptitle("Problem 3.4b")
plt.savefig("../../figures/Chapter3/Problem3_4b", dpi=300)