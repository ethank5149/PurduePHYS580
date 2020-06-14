########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 6                                                                            #
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
from lib.FindRoot import secant
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
m = 1
g = 9.81
l = 1

def f(t, X, dX, q):
    return np.array([-(g/l)*X[0]-q*dX[0],])

t = np.linspace(0,10,2000)
x1, dx1 = eulercromer(partial(f, q=1),[1,],[0],t)
x2, dx2 = eulercromer(partial(f, q=5),[1,],[0],t)
x3, dx3 = eulercromer(partial(f, q=10),[1,],[0],t)


fig, ax = plt.subplots(1, 1)
ax.plot(t, x1[0], label=r"$q = 1$")
ax.plot(t, x2[0], label=r"$q = 5$")
ax.plot(t, x3[0], label=r"$q = 10$")
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [m]")
ax.set_title("Varying Damping Coefficients")
plt.legend()
plt.suptitle("Problem 3.6")
plt.savefig("../../figures/Chapter3/Problem3_6", dpi=300)


infinity = 1000
ninfinity = 10*infinity
t = np.linspace(0,infinity,ninfinity)

# From trial and error
a, c = 5.283, 5.284
a, c = 5, 7
tol = 0.000001
MAX_ITERATIONS = 100

def F(x):
    return np.sum(np.where(eulercromer(partial(f, q=x), [1, ], [0], t)[0]<0,1,0))

def modified_bisection(a, c):
    f_a = F(a)
    f_c = F(c)
    b = 0.5 * (a + c)

    if (f_a>0 and f_c != 0) or (f_a == 0 and f_c == 0):
        print("Invalid Bound")
        return None

    if f_a == 0 and f_c > 0:
        a, c = c, a

    for iteration in range(MAX_ITERATIONS):
        b = 0.5 * (a + c)
        f_b = F(b)

        if abs(c - a) < tol:
            return b

        if f_b > 0:
            a = b
        elif f_b == 0:
            c = b

        f_a = F(a)
        f_c = F(c)
        print(f"{a}, {b}, {c} : {f_a}, {f_b}, {f_c}")
        if not (f_a>0 and f_c == 0):
            print("Failed")
            print("Best Guess: ", b)
            return None
    return b


q_crit = modified_bisection(a, c)
print(q_crit-2*np.sqrt(g/l))

soln = eulercromer(partial(f, q=q_crit),[1,],[0],t)
fig, ax = plt.subplots(1, 1)
ax.plot(t[:int(ninfinity/1000)], soln[0][0][:int(ninfinity/1000)], label=rf"$q = {q_crit}$")
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [m]")
ax.set_title("Critical Damping")
plt.legend()
plt.suptitle("Problem 3.6b")
plt.savefig("../../figures/Chapter3/Problem3_6b", dpi=300)
