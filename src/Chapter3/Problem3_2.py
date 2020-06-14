########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 2                                                                            #
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
g = 9.81
l = 1
m = 1
Omega = np.sqrt(g/l)
T = 2*np.pi/Omega


def f(t, X, dX):
    return np.array([-(g/l)*np.sin(X[0]),])


def f(t, X, dX):
    return np.array([-(g/l)*X[0],])


def f_rk4(t, X):
    return np.array([X[1],-(g/l)*X[0]])


def E(t, X, dX):
    return 0.5*m*l**2*dX[0]**2+m*g*l*(1-np.cos(X[0]))


def E_rk4(t, X):
    return 0.5*m*l**2*X[1]**2+m*g*l*(1-np.cos(X[0]))


def diff_E(t, X, dX):
    true_E = 0.5*m*l**2*dX[0,0]**2+m*g*l*(1-np.cos(X[0,0]))
    return 100*(0.5*m*l**2*dX[0]**2+m*g*l*(1-np.cos(X[0])) - true_E)/true_E


def diff_E_rk4(t, X):
    true_E = 0.5*m*l**2*X[1,0]**2+m*g*l*(1-np.cos(X[0,0]))
    return 100*(0.5*m*l**2*X[1]**2+m*g*l*(1-np.cos(X[0])) - true_E)/true_E


num_periods = 6
t = np.linspace(0,num_periods*T,2000)
x_rk4 = rk4(f_rk4,[np.pi/3,0],t)
x, dx = eulercromer(f,[np.pi/3,],[0.,],t)
fig, ax = plt.subplots(1, 1)
ax.plot(t, diff_E_rk4(t,x_rk4), label="RK4")
ax.plot(t, diff_E(t,x, dx), label="Euler-Cromer")
for i in range(num_periods):
    plt.axvline(x=(i+1)*T, color='k', linestyle='--')
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel("Percent Error")
ax.set_title("Energy Percent Error")
plt.legend()
plt.suptitle("Problem 3.2a")
plt.savefig("../../figures/Chapter3/Problem3_2a", dpi=300)


fig, ax = plt.subplots(1, 1)
ax.plot(t, diff_E_rk4(t,x_rk4)-diff_E(t,x, dx))
for i in range(num_periods):
    plt.axvline(x=(i+1)*T, color='k', linestyle='--')
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel("% Error")
ax.set_title("RK4 % Error - Euler-Cromer % Error")
plt.suptitle("Problem 3.2b")
plt.savefig("../../figures/Chapter3/Problem3_2b", dpi=300)