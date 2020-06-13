########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 8                                                                            #
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
from lib.DSolve import euler
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
earth = Constants.Earth()


# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
x0 = y0 = 0
v0 = 10000


def rhs_varying_g(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_ref*v*X[2]/m, -earth.GM/(earth.radius+X[1])**2-B2_ref*v*X[3]/m])


def rhs_constant_g(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_ref*v*X[2]/m, -g-B2_ref*v*X[3]/m])


def terminate(X):
    return X[1] < 0


def f_ics(theta):
    return [x0, y0, v0*np.cos(np.pi*theta/180), v0*np.sin(np.pi*theta/180)]



angles = (30, 45, 60)
t = np.linspace(0,400,40000)

# Part A
# Varying vs Constant Gravity
soln00 = euler(rhs_varying_g, f_ics(angles[0]), t, terminate=terminate)
soln01 = euler(rhs_varying_g, f_ics(angles[1]), t, terminate=terminate)
soln02 = euler(rhs_varying_g, f_ics(angles[2]), t, terminate=terminate)
soln10 = euler(rhs_constant_g, f_ics(angles[0]), t, terminate=terminate)
soln11 = euler(rhs_constant_g, f_ics(angles[1]), t, terminate=terminate)
soln12 = euler(rhs_constant_g, f_ics(angles[2]), t, terminate=terminate)

minx0 = min(np.size(soln00[0]), np.size(soln10[0]))
minx1 = min(np.size(soln01[0]), np.size(soln11[0]))
minx2 = min(np.size(soln02[0]), np.size(soln12[0]))
miny0 = min(np.size(soln00[1]), np.size(soln10[1]))
miny1 = min(np.size(soln01[1]), np.size(soln11[1]))
miny2 = min(np.size(soln02[1]), np.size(soln12[1]))
min0 = min(minx0, miny0)
min1 = min(minx1, miny1)
min2 = min(minx2, miny2)

# Plotting
fig, ax = plt.subplots(1, 1)
ax.plot((soln00[0, :min0]-soln10[0, :min0])/1000, (soln00[1, :min0]-soln10[1, :min0])/1000,
        label=rf"$\theta = 30^{{\circ}}$")
ax.plot((soln01[0, :min1]-soln11[0, :min1])/1000, (soln01[1, :min1]-soln11[1, :min1])/1000,
        label=rf"$\theta = 45^{{\circ}}$")
ax.plot((soln02[0, :min2]-soln12[0, :min2])/1000, (soln02[1, :min2]-soln12[1, :min2])/1000,
        label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8a")
plt.savefig("../../figures/Chapter2/Problem2_8a", dpi=300)

# Part B
fig, ax = plt.subplots(1, 1)

ax.plot(t[:minx0], (soln00[0, :minx0]-soln10[0, :minx0])/1000, label=rf"$\theta = 30^{{\circ}}$")
ax.plot(t[:minx1], (soln01[0, :minx1]-soln11[0, :minx1])/1000, label=rf"$\theta = 45^{{\circ}}$")
ax.plot(t[:minx2], (soln02[0, :minx2]-soln12[0, :minx2])/1000, label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("x [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8b")
plt.savefig("../../figures/Chapter2/Problem2_8b", dpi=300)

# Part C
fig, ax = plt.subplots(1, 1)

ax.plot(t[:miny0], (soln00[1, :miny0]-soln10[1, :miny0])/1000, label=rf"$\theta = 30^{{\circ}}$")
ax.plot(t[:miny1], (soln01[1, :miny1]-soln11[1, :miny1])/1000, label=rf"$\theta = 45^{{\circ}}$")
ax.plot(t[:miny2], (soln02[1, :miny2]-soln12[1, :miny2])/1000, label=rf"$\theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("t [s]")
ax.set_ylabel("y [km]")
ax.set_title("Varying - Constant")
plt.suptitle("Problem 2.8c")
plt.savefig("../../figures/Chapter2/Problem2_8c", dpi=300)
