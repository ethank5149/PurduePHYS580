########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 16                                                                           #
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
from lib.DSolve import odeint
import numpy as np
from functools import partial
from matplotlib import pyplot as plt


# Global Definitions
m = 1
l = g = 9.8
q = 0.5
Omega_D = 2/3


def f(t, X, dX, F_D):
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0]+F_D*np.sin(Omega_D*t), ])


def constrain_domain(x, dx):
    if x[0] < -np.pi:
        x[0] += 2 * np.pi
    elif x[0] > np.pi:
        x[0] -= 2 * np.pi
    return x, dx


num_points = 1000
point_density = 1000
t = np.linspace(0, 2*np.pi*num_points/Omega_D, point_density*num_points)

theta0 = 0.2
dtheta0 = 0
F_D_1 = 1.16
F_D_2 = 1.2
F_D_3 = 1.24
F_D_4 = 1.26

# x1, dx1 = odeint(partial(f,F_D=F_D_1),([theta0,],[dtheta0,]),t,method='verlet',fargs=(constrain_domain,))
x2, dx2 = odeint(partial(f,F_D=F_D_2),([theta0,],[dtheta0,]),t,method='verlet',fargs=(constrain_domain,))
# x3, dx3 = odeint(partial(f,F_D=F_D_3),([theta0,],[dtheta0,]),t,method='verlet',fargs=(constrain_domain,))
# x4, dx4 = odeint(partial(f,F_D=F_D_4),([theta0,],[dtheta0,]),t,method='verlet',fargs=(constrain_domain,))
#
# fig, ax = plt.subplots(1, 1)
# ax.scatter(x1[0][::point_density],dx1[0][::point_density],s=0.1,label=rf"$F_D={F_D_1}$")
# ax.scatter(x2[0][::point_density],dx2[0][::point_density],s=0.1,label=rf"$F_D={F_D_2}$")
# ax.scatter(x3[0][::point_density],dx3[0][::point_density],s=0.1,label=rf"$F_D={F_D_3}$")
# ax.scatter(x4[0][::point_density],dx4[0][::point_density],s=0.1,label=rf"$F_D={F_D_4}$")
#
# ax.grid()
# ax.set_xlabel(r"$\theta\,\,[rad]$")
# ax.set_ylabel(r"$\dot{\theta}\,\,[rad/s]$")
# ax.legend()
# ax.set_title("Poincare Plot")
#
# plt.suptitle(r"Problem 3.16")
# plt.savefig("../../figures/Chapter3/Problem3_16", dpi=300)

plt.cla()
plt.scatter(x2[0], dx2[0], s=0.025, label=rf"$F_D={F_D_2}$")
plt.show()
