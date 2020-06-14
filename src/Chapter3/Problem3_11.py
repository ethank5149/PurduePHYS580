########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 11                                                                           #
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
from lib.DSolve import rk4_pendulum, eulercromer_pendulum
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
m = 1
l = g = 9.81
q = 0.5
Omega_D = 2/3


def f(t, X, dX, F_D):
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0]+F_D*np.sin(Omega_D*t),])


def f_rk4(t, X, F_D):
    return np.array([-(g/l)*np.sin(X[0])-q*X[1]+F_D*np.sin(Omega_D*t),])


def energy(t, X, dX):
    return 0.5*m*l*dX**2+m*g*l*(1-np.cos(X))


def energy_added(t, X, dX, F_D):
    return ((F_D/Omega_D)*np.cos(Omega_D*t))**2/(2*m)


def energy_lost(t, X, dX):
    return (q*X)**2/(2*m)


t = np.linspace(0,60,6000)  # dt = 0.01
# x1, dx1 = eulercromer_pendulum(partial(f,F_D=0.1),[0.2,],[0],t)
# x2, dx2 = eulercromer_pendulum(partial(f,F_D=0.5),[0.2,],[0],t)
# x3, dx3 = eulercromer_pendulum(partial(f,F_D=0.99),[0.2,],[0],t)

x1 = rk4_pendulum(partial(f_rk4,F_D=0.1),[0.2,0],t)
x2 = rk4_pendulum(partial(f_rk4,F_D=0.5),[0.2,0],t)
x3 = rk4_pendulum(partial(f_rk4,F_D=0.99),[0.2,0],t)

fig, ax = plt.subplots(1, 1)

# ax.plot(t, energy(t,x1[0],dx1[0]) - energy(t,x1[0][0],dx1[0][0]) ,label=r"$F_D=0.1$")
# ax.plot(t, energy(t,x2[0],dx2[0]) - energy(t,x2[0][0],dx2[0][0]) ,label=r"$F_D=0.5$")
# ax.plot(t, energy(t,x3[0],dx3[0]) - energy(t,x3[0][0],dx3[0][0]) , label=r"$F_D=0.99$")

ax.plot(t, energy(t,x1[0],x1[1]) - energy(t,x1[0][0],x1[1][0]),label=r"$F_D=0.1$")
ax.plot(t, energy(t,x2[0],x2[1]) - energy(t,x2[0][0],x2[1][0]), label=r"$F_D=0.5$")
ax.plot(t, energy(t,x3[0],x3[1]) - energy(t,x3[0][0],x3[1][0]), label=r"$F_D=0.99$")

ax.grid()
ax.set_xlabel(r"$t\,\,[s]$")
ax.set_ylabel(r"$Energy\,\,[J]$")
ax.legend()
ax.set_title("Energy Change vs. Time")

plt.suptitle("Problem 3.11")
plt.savefig("../../figures/Chapter3/Problem3_11", dpi=300)

