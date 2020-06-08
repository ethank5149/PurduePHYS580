########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 5                                                                            #
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
from lib.NDSolveSystem import ODE, SymplecticODE
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


def rhs(t, X, m, P, C, rho, A,F0):
    if F0*X[0] < P:
        return np.array([F0/m - 0.5 * C * rho * A * X[0] ** 2 / m, ])
    else:
        return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m,])


def rhs_old(t, X, m, P, C, rho, A):
    return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m,])


power = 400
mass = 70
v_initial = 4
drag_coeff = 0.5
v_star = 7
F_initial = power/v_star
area = 0.33
air_density = 1.225

curried_rhs = partial(rhs, m=mass,P=power,C=drag_coeff,rho=air_density,A=area,F0=F_initial)
curried_rhs_old = partial(rhs_old, m=mass,P=power,C=drag_coeff,rho=air_density,A=area)
ic = np.array([v_initial,])  # Initial Condition

sim1 = ODE(curried_rhs, ic, ti=0, dt=0.01, tf=50)
sim2 = ODE(curried_rhs_old, ic, ti=0, dt=0.01, tf=50)

sim1.run()
sim2.run()

# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(sim1.t_series, sim1.X_series[:,0], label=f"New")
ax.plot(sim2.t_series, sim2.X_series[:,0], label=f"Old")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")


plt.suptitle("Problem 2.5")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_5",dpi=300)
