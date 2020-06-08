########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 4                                                                            #
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
# Including
from lib.NDSolveSystem import ODE
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
w = 0.411  # Shoulder Width [m]
air = Constants.Air()
water = Constants.Water()


def rhs(t, X, m, P, C, rho, theta, A):
    return np.array([P/(m*X[0])-g*np.sin(theta)-0.5*C*rho*A*X[0]**2/m,])


P = 400
m = 70
v0 = 4
C = 0.5
angle_deg = 10
angle = angle_deg*np.pi/180
A = 0.33

curried_rhs_flat = partial(rhs, m=m, P=P, C=C, rho=air.density, theta=0, A=A)
curried_rhs_incline = partial(rhs, m=m, P=P, C=C, rho=air.density, theta=angle, A=A)
curried_rhs_decline = partial(rhs, m=m, P=P, C=C, rho=air.density, theta=-angle, A=A)

ic = np.array([v0, ])

sim1 = ODE(curried_rhs_flat, ic, ti=0, dt=0.01, tf=100)
sim2 = ODE(curried_rhs_incline, ic, ti=0, dt=0.01, tf=100)
sim3 = ODE(curried_rhs_decline, ic, ti=0, dt=0.01, tf=100)

sim1.run()
sim2.run()
sim3.run()

# Plotting
fig, ax = plt.subplots(1,1)

ax.plot(sim1.t_series, sim1.X_series[:,0], label=rf"$0^{{\circ}}$ Grade")
ax.plot(sim2.t_series, sim2.X_series[:,0], label=rf"${angle_deg}^{{\circ}}$ Grade")
ax.plot(sim3.t_series, sim3.X_series[:,0], label=rf"$-{angle_deg}^{{\circ}}$ Grade")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")

plt.suptitle("Problem 2.4")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_4",dpi=300)
