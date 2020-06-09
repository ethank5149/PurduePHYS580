########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 7                                                                            #
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


def B2(T):
    return B2_ref*(T_0/T)**alpha


def B2_eff(T,rho):
    return (rho/air.density)*B2(T)


def rhs_adiabatic(t, X, T):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(T,rho_adiabatic(X[1]))*v*X[2]/m, -g-B2_eff(T,rho_adiabatic(X[1]))*v*X[3]/m])


def rhs_isothermal(t, X, T):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(T,rho_isothermal(X[1]))*v*X[2]/m, -g-B2_eff(T,rho_isothermal(X[1]))*v*X[3]/m])


def terminate(X):
    return X[1] < 0


def deg_to_rad(theta):
    return np.pi*theta/180


x0 = y0 = 0
v0 = 700
angles_deg = (30, 45, 60)

angles_rad = tuple([deg_to_rad(angle) for angle in angles_deg])
ics = tuple([np.array([x0, y0, v0*np.cos(theta), v0*np.sin(theta)]) for theta in angles_rad])

# Part A
# Adiabatic vs Isothermal, Normal Temperature
sims_adiabatic = tuple([ODE(partial(rhs_adiabatic, T=T_0), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_adiabatic:
    sim.run()
sims_isothermal = tuple([ODE(partial(rhs_isothermal, T=T_0), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_isothermal:
    sim.run()

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot(sims_isothermal[0].X_series[:,0]/1000, sims_isothermal[0].X_series[:,1]/1000,'r-',
        label=rf"Isothermal, $\theta = 30^{{\circ}}$")
ax.plot(sims_isothermal[1].X_series[:,0]/1000, sims_isothermal[1].X_series[:,1]/1000,'b-',
        label=rf"Isothermal, $\theta = 45^{{\circ}}$")
ax.plot(sims_isothermal[2].X_series[:,0]/1000, sims_isothermal[2].X_series[:,1]/1000,'k-',
        label=rf"Isothermal, $\theta = 60^{{\circ}}$")
ax.plot(sims_adiabatic[0].X_series[:,0]/1000, sims_adiabatic[0].X_series[:,1]/1000,'r--',
        label=rf"$Adiabatic, \theta = 30^{{\circ}}$")
ax.plot(sims_adiabatic[1].X_series[:,0]/1000, sims_adiabatic[1].X_series[:,1]/1000,'b--',
        label=rf"$Adiabatic, \theta = 45^{{\circ}}$")
ax.plot(sims_adiabatic[2].X_series[:,0]/1000, sims_adiabatic[2].X_series[:,1]/1000,'k--',
        label=rf"$Adiabatic, \theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Adiabatic vs. Isothermal Model")
plt.suptitle("Problem 2.7a")
plt.savefig("../../figures/Chapter2/Problem2_7a",dpi=300)

# Part B
# Adiabatic, Hot vs Cold
sims_cold = tuple([ODE(partial(rhs_adiabatic, T=T_cold), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_cold:
    sim.run()
sims_hot = tuple([ODE(partial(rhs_adiabatic, T=T_hot), ic, ti=0, dt=0.01, tf=200,terminate=terminate) for ic in ics])
for sim in sims_hot:
    sim.run()

# Plotting
fig, ax = plt.subplots(1,1)
ax.plot(sims_cold[0].X_series[:,0]/1000, sims_cold[0].X_series[:,1]/1000,'r-',
        label=rf"Cold, $\theta = 30^{{\circ}}$")
ax.plot(sims_cold[1].X_series[:,0]/1000, sims_cold[1].X_series[:,1]/1000,'b-',
        label=rf"Cold, $\theta = 45^{{\circ}}$")
ax.plot(sims_cold[2].X_series[:,0]/1000, sims_cold[2].X_series[:,1]/1000,'k-',
        label=rf"Cold, $\theta = 60^{{\circ}}$")
ax.plot(sims_hot[0].X_series[:,0]/1000, sims_hot[0].X_series[:,1]/1000,'r--',
        label=rf"$Hot, \theta = 30^{{\circ}}$")
ax.plot(sims_hot[1].X_series[:,0]/1000, sims_hot[1].X_series[:,1]/1000,'b--',
        label=rf"$Hot, \theta = 45^{{\circ}}$")
ax.plot(sims_hot[2].X_series[:,0]/1000, sims_hot[2].X_series[:,1]/1000,'k--',
        label=rf"$Hot, \theta = 60^{{\circ}}$")

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_title("Adiabatic Model, Hot vs. Cold Temperatures")
plt.suptitle("Problem 2.7b")
plt.savefig("../../figures/Chapter2/Problem2_7b",dpi=300)

# Part C
def scan_max_range_temp(num_angles, num_temps):
    temperatures = np.linspace(T_cold, T_hot, num_temps)
    angles = np.linspace(np.pi/6, np.pi/3, num_angles)
    initial_conditions = [np.array([x0, y0, v0 * np.cos(angle), v0 * np.sin(angle)]) for angle in angles]
    functions = [partial(rhs_adiabatic, T=t) for t in temperatures]
    max_ranges = []
    for function in functions:
        simulations = [ODE(function, ic, ti=0, dt=0.01, tf=200, terminate=terminate) for ic in initial_conditions]
        for simulation in simulations:
            simulation.run()
        max_ranges.append(max([simulation.X_series[-1, 0] for simulation in simulations]))
    return temperatures, np.array(max_ranges)


temps, ranges = scan_max_range_temp(25, 25)
# Plotting
fig, ax = plt.subplots(1, 1)
ax.plot(temps,ranges)
ax.grid()
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Range [km]")
ax.set_title("Maximum Range Vs. Temperature")
plt.suptitle("Problem 2.7c")
plt.savefig("../../figures/Chapter2/Problem2_7c", dpi=300)
