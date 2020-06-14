########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 9                                                                            #
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
from lib.DSolve import eulercromer
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
m = 1
g = 9.81
q = 0.1
l = 1
theta0 = 0.5
omega = np.sqrt(g/l)


def f(t, X, dX):
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0],])

# def f(t, X, dX):
#     return np.array([-(g/l)*X[0]-q*dX[0],])


A = theta0
tc = 19.845

t = np.linspace(0,100,2000)
x, dx = eulercromer(f,[theta0,],[0],t)
model = A*np.cos(omega*t)*np.exp(-t/tc)
print(np.sum(np.absolute(x[0]-model)))

fig, ax = plt.subplots(1, 1)
ax.plot(t, x[0]-model, label='Error')
ax.plot(t, x[0], label="Experiment")
ax.plot(t,model,'--',label=r'$A\cos(\omega t)e^{-\frac{t}{\tau}}$')
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [m]")
plt.legend()
plt.suptitle("Problem 3.9")
# plt.show()
plt.savefig("../../figures/Chapter3/Problem3_9", dpi=300)

