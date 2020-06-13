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
from lib.DSolve import euler
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
w = 0.411  # Shoulder Width [m]
P = 400  # Power [W]
m = 70  # Mass [kg]
v0 = 4  # Initial velocity [m/s]
C = 0.5  # Drag coefficient
angle_deg = 10
angle = angle_deg*np.pi/180
A = 0.33
air = Constants.Air()
rho = air.density


def rhs(t, X, theta):
    return np.array([P/(m*X[0])-g*np.sin(theta)-0.5*C*rho*A*X[0]**2/m, ])


curried_rhs_flat = partial(rhs, theta=0)
curried_rhs_incline = partial(rhs, theta=angle)
curried_rhs_decline = partial(rhs, theta=-angle)
t = np.linspace(0, 100, 10000)

y1 = euler(curried_rhs_flat, [v0, ], t)
y2 = euler(curried_rhs_incline, [v0, ], t)
y3 = euler(curried_rhs_decline, [v0, ], t)

# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(t, y1[0], label=rf"$0^{{\circ}}$ Grade")
ax.plot(t, y2[0], label=rf"${angle_deg}^{{\circ}}$ Grade")
ax.plot(t, y3[0], label=rf"$-{angle_deg}^{{\circ}}$ Grade")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")

plt.suptitle("Problem 2.4")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_4", dpi=300)
