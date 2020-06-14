########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 13                                                                           #
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
    return np.array([-(g/l)*np.sin(X[0])-q*dX[0]+F_D*np.sin(Omega_D*t),])


def constrain_domain(x, dx):
    if x[0] < -np.pi:
        x[0] += 2 * np.pi
    elif x[0] > np.pi:
        x[0] -= 2 * np.pi
    return x, dx


t = np.linspace(0,200,100000)
theta0 = 0.2
dtheta0 = 0.00000001
# x1, dx1 = odeint(partial(f,F_D=1.2),([theta0,],[0,]),t,method='verlet',fargs=(constrain_domain,))
# x2, dx2 = odeint(partial(f,F_D=1.2),([theta0+dtheta0,],[0,]),t,method='verlet',fargs=(constrain_domain,))
x1, dx1 = odeint(partial(f,F_D=1.2),([theta0,],[0,]),t,method='verlet')
x2, dx2 = odeint(partial(f,F_D=1.2),([theta0+dtheta0,],[0,]),t,method='verlet')
fig, ax = plt.subplots(1, 1)

ax.plot(t,np.log(np.absolute(x2[0]-x1[0])),label=r"$\ln{\left|\Delta\theta\right|}$")
ax.set_title(rf"$\Delta\theta_0={dtheta0}$")
ax.grid()
ax.set_xlabel(r"$t\,\,[s]$")
ax.set_ylabel(r"$\ln{\left|\Delta\theta\right|}$")
ax.legend()

plt.suptitle(r"Problem 3.13")
plt.savefig("../../figures/Chapter3/Problem3_13", dpi=300)