########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 11                                                                           #
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
from lib.Constants import Earth, Air
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
earth = Earth()
air = Air()


# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
a = 6.5e-3  # Adiabatic model parameter [K/m]
alpha = 2.5  # Adiabatic model parameter
y_0 = 1.0e4  # k_B*T/mg [m]
T_0 = 300  # Adiabatic model sea level temperature [K]
x0 = y0 = 0
v0 = 1000



def B2_eff(y):
    return B2_ref*(1-a*y/T_0)**alpha


def rhs(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(X[1])*v*X[2]/m, -earth.GM/(earth.radius+X[1])**2-B2_eff(X[1])*v*X[3]/m])


def terminate(X, dist, elev):
    return X[0]>0 and (X[1] < 0 and X[0]<dist) or (X[1] < elev and X[0]>dist)


def f_ics(v0, theta):
    return [x0, y0, v0*np.cos(np.pi*theta/180), v0*np.sin(np.pi*theta/180)]


angles = (45, 55, 65)
distance_to_cliff = 10000
elevation_of_cliff = 10000
t = np.linspace(0, 400, 40000)
velocities = np.linspace(200, 800, 250)


def calc_max_range(v, angle, elev):
    soln = euler(rhs, f_ics(v, angle), t, terminate=partial(terminate,dist=distance_to_cliff, elev=elev))
    return soln[0,-1]


def calc_v_senstivity(v, angle, elev):
    control = calc_max_range(v, angle, elev)
    one_percent_diff = calc_max_range(0.99*v, angle, elev)
    return control - one_percent_diff


sensitivity_45 = np.asarray([calc_v_senstivity(v,45,elevation_of_cliff) for v in velocities])
sensitivity_55 = np.asarray([calc_v_senstivity(v,55,elevation_of_cliff) for v in velocities])
sensitivity_65 = np.asarray([calc_v_senstivity(v,65,elevation_of_cliff) for v in velocities])

fig, ax = plt.subplots(1, 1)
ax.plot(velocities, sensitivity_45, label=rf"$\theta = 45^{{\circ}}$")
ax.plot(velocities, sensitivity_55, label=rf"$\theta = 55^{{\circ}}$")
ax.plot(velocities, sensitivity_65, label=rf"$\theta = 65^{{\circ}}$")
ax.legend()
ax.grid()
ax.set_xlabel("v [m/s]")
ax.set_ylabel("Range Difference [m]")
ax.set_title("Cliff Elevation = 10 [km]")
plt.suptitle("Problem 2.11a")
plt.savefig("../../figures/Chapter2/Problem2_11a", dpi=300)


sensitivity_45 = np.asarray([calc_v_senstivity(v,45,-elevation_of_cliff) for v in velocities])
sensitivity_55 = np.asarray([calc_v_senstivity(v,55,-elevation_of_cliff) for v in velocities])
sensitivity_65 = np.asarray([calc_v_senstivity(v,65,-elevation_of_cliff) for v in velocities])

fig, ax = plt.subplots(1, 1)
ax.plot(velocities, sensitivity_45, label=rf"$\theta = 45^{{\circ}}$")
ax.plot(velocities, sensitivity_55, label=rf"$\theta = 55^{{\circ}}$")
ax.plot(velocities, sensitivity_65, label=rf"$\theta = 65^{{\circ}}$")
ax.legend()
ax.grid()
ax.set_xlabel("v [m/s]")
ax.set_ylabel("Range Difference [m]")
ax.set_title("Cliff Elevation = -10 [km]")
plt.suptitle("Problem 2.11b")
plt.savefig("../../figures/Chapter2/Problem2_11b", dpi=300)