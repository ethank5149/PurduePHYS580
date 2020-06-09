########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 9                                                                            #
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
from lib.NDSolveSystem import ODE
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
air = Constants.Air()
earth = Constants.Earth()

# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
a = 6.5e-3  # Adiabatic model parameter [K/m]
alpha = 2.5  # Adiabatic model parameter
y_0 = 1.0e4  # k_B*T/mg [m]
T_0 = 300  # Adiabatic model sea level temperature [K]
T_hot = 310.9278  # "Hot summer day" temperature ~ 100 [F]
T_cold = 266.4833  # "Cold winter day" temperature ~ 20 [F]


def rho_isothermal(y):
    return air.density*np.exp(-y/y_0)


def rho_adiabatic(y):
    return air.density*(1-a*y/T_0)**alpha


def B2_eff(rho):
    return (rho/air.density)*B2_ref


def rhs_adiabatic(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(rho_adiabatic(X[1]))*v*X[2]/m,-earth.GM/(earth.radius+X[1])**2-B2_eff(rho_adiabatic(X[1]))*v*X[3]/m])


def rhs_isothermal(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(rho_isothermal(X[1]))*v*X[2]/m,-earth.GM/(earth.radius+X[1])**2-B2_eff(rho_isothermal(X[1]))*v*X[3]/m])


def rhs(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_ref*v*X[2]/m,-earth.GM/(earth.radius+X[1])**2-B2_ref*v*X[3]/m])


def terminate(X):
    return X[1] < 0


def deg_to_rad(theta):
    return np.pi*theta/180


x0 = y0 = 0
v0 = 700
angles_deg = (35, 45)

angles_rad = tuple([deg_to_rad(angle) for angle in angles_deg])
ics = tuple([np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)]) for theta in angles_rad])

# Part A
# Adiabatic vs Isothermal, Normal Temperature
sims_adiabatic = tuple([ODE(rhs_adiabatic, ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_adiabatic:
    sim.run()
sims_isothermal = tuple([ODE(rhs_isothermal, ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_isothermal:
    sim.run()
sims = tuple([ODE(rhs, ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims:
    sim.run()

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot(sims_isothermal[0].X_series[:,0]/1000, sims_isothermal[0].X_series[:,1]/1000,'r-.',
        label=rf"Isothermal, $\theta = 35^{{\circ}}$")
ax.plot(sims_isothermal[1].X_series[:,0]/1000, sims_isothermal[1].X_series[:,1]/1000,'b-.',
        label=rf"Isothermal, $\theta = 45^{{\circ}}$")
ax.plot(sims_adiabatic[0].X_series[:,0]/1000, sims_adiabatic[0].X_series[:,1]/1000,'r--',
        label=rf"$Adiabatic, \theta = 35^{{\circ}}$")
ax.plot(sims_adiabatic[1].X_series[:,0]/1000, sims_adiabatic[1].X_series[:,1]/1000,'b--',
        label=rf"$Adiabatic, \theta = 45^{{\circ}}$")
ax.plot(sims[0].X_series[:,0]/1000, sims[0].X_series[:,1]/1000,'r-',
        label=rf"$Control, \theta = 35^{{\circ}}$")
ax.plot(sims[1].X_series[:,0]/1000, sims[1].X_series[:,1]/1000,'b-',
        label=rf"$Control, \theta = 45^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Adiabatic vs. Isothermal Model")
plt.suptitle("Problem 2.9a")
plt.savefig("../../figures/Chapter2/Problem2_9a",dpi=300)


def scan_max_range(rhs,start_angle,end_angle,num_angles):
    angles = np.linspace(start_angle,end_angle, num_angles)
    initial_conditions = [np.array([x0, y0, v0 * np.cos(np.pi*angle/180), v0 * np.sin(np.pi*angle/180)]) for angle in angles]
    simulations = [ODE(rhs, ic, ti=0, dt=0.01, tf=200, terminate=terminate) for ic in initial_conditions]
    for simulation in simulations:
        simulation.run()
    return angles, np.array([simulation.X_series[-1, 0] for simulation in simulations])


# Part B
angles_adiabatic, ranges_adiabatic = scan_max_range(rhs_adiabatic,30,60,75)
max_range_adiabatic = max(list(ranges_adiabatic))
max_range_angle_adiabatic = angles_adiabatic[list(ranges_adiabatic).index(max_range_adiabatic)]

fig, ax = plt.subplots(1, 1)
ax.plot(angles_adiabatic,ranges_adiabatic/1000)
plt.axvline(x=max_range_angle_adiabatic,color='k')
plt.axhline(y=max_range_adiabatic/1000,color='k')
ax.grid()
ax.set_xlabel("Angle [deg]")
ax.set_ylabel("Range [km]")
ax.set_title("Maximum Range Vs. Angle - Adiabatic Model")
plt.suptitle("Problem 2.9b")
plt.savefig("../../figures/Chapter2/Problem2_9b", dpi=300)

# Part C
angles_isothermal, ranges_isothermal = scan_max_range(rhs_isothermal,30,60,75)
max_range_isothermal = max(list(ranges_isothermal))
max_range_angle_isothermal = angles_isothermal[list(ranges_isothermal).index(max_range_isothermal)]

fig, ax = plt.subplots(1, 1)
ax.plot(angles_isothermal,ranges_isothermal/1000)
plt.axvline(x=max_range_angle_isothermal,color='k')
plt.axhline(y=max_range_isothermal/1000,color='k')
ax.grid()
ax.set_xlabel("Angle [deg]")
ax.set_ylabel("Range [km]")
ax.set_title("Maximum Range Vs. Angle - Isothermal Model")
plt.suptitle("Problem 2.9c")
plt.savefig("../../figures/Chapter2/Problem2_9c", dpi=300)