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
from lib.NDSolveSystem import ODE
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
earth = Constants.Earth()
air = Constants.Air()


# Global Definitions
g = 9.81  # Gravitational acceleration [m/s^2]
m = 47.5  # Mass [kg] of a 10 cm lead sphere
B2_ref = 4.0e-5 * m  # Air resistance coefficient
a = 6.5e-3  # Adiabatic model parameter [K/m]
alpha = 2.5  # Adiabatic model parameter
y_0 = 1.0e4  # k_B*T/mg [m]
T_0 = 300  # Adiabatic model sea level temperature [K]



def rho_adiabatic(y):
    return air.density*(1-a*y/T_0)**alpha


def B2_eff(rho):
    return (rho/air.density)*B2_ref


def rhs(t, X):
    v = np.sqrt(X[2]**2+X[3]**2)
    return np.array([X[2], X[3], -B2_eff(rho_adiabatic(X[1]))*v*X[2]/m,-earth.GM/(earth.radius+X[1])**2-B2_eff(rho_adiabatic(X[1]))*v*X[3]/m])


def terminate(X):
    return X[0]>0 and (X[1] < 0 and X[0]<distance_to_cliff) or (X[1] < elevation_of_cliff and X[0]>distance_to_cliff)


x0 = y0 = 0
v0 = 1000
angles = (30,35,40, 45,50,55, 60)
distance_to_cliff = 10000
elevation_of_cliff = 10000

# Part A
ics = tuple([np.array([x0, y0, v0*np.cos(np.pi*theta/180), v0*np.sin(np.pi*theta/180)]) for theta in angles])
sims = tuple([ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate) for ic in ics])
for sim in sims:
    sim.run()

fig, ax = plt.subplots(1, 1)
for i,sim in enumerate(sims):
    ax.plot(sim.X_series[0]/1000, sim.X_series[1]/1000, label=rf"$\theta = {angles[i]}^{{\circ}}$")

if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff/1000-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff/1000,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff/1000),
                                (elevation_of_cliff/1000-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff/1000, 0),(plt.xlim()[1]-distance_to_cliff/1000),
                               elevation_of_cliff/1000,color='green')
    ax.add_patch(cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10a")
plt.savefig("../../figures/Chapter2/Problem2_10a", dpi=300)


# Part B
x0 = y0 = 0
v0 = 1000
angles = (30,35,40, 45,50,55, 60)
distance_to_cliff = 10000
elevation_of_cliff = -10000

ics = tuple([np.array([x0, y0, v0*np.cos(np.pi*theta/180), v0*np.sin(np.pi*theta/180)]) for theta in angles])
sims = tuple([ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate) for ic in ics])
for sim in sims:
    sim.run()

fig, ax = plt.subplots(1, 1)
for i,sim in enumerate(sims):
    ax.plot(sim.X_series[0]/1000, sim.X_series[1]/1000, label=rf"$\theta = {angles[i]}^{{\circ}}$")

if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff/1000-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff/1000,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff/1000),
                                (elevation_of_cliff/1000-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff/1000, 0),(plt.xlim()[1]-distance_to_cliff/1000),
                               elevation_of_cliff/1000,color='green')
    ax.add_patch(cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10b")
plt.savefig("../../figures/Chapter2/Problem2_10b", dpi=300)

# Part C
x0 = y0 = 0
v0 = 1000
angle = 60
velocities = (600,700,800,833.6379016603621, 900,1000)
distance_to_cliff = 10000
elevation_of_cliff = 10000

ics = tuple([np.array([x0, y0, v*np.cos(np.pi/3), v*np.sin(np.pi/3)]) for v in velocities])
sims = tuple([ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate) for ic in ics])
for sim in sims:
    sim.run()

fig, ax = plt.subplots(1, 1)
for i,sim in enumerate(sims):
    ax.plot(sim.X_series[0]/1000, sim.X_series[1]/1000, label=rf"$v_0 = {velocities[i]}$")

if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff/1000-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff/1000,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff/1000),
                                (elevation_of_cliff/1000-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff/1000, 0),(plt.xlim()[1]-distance_to_cliff/1000),
                               elevation_of_cliff/1000,color='green')
    ax.add_patch(cliff)

ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.10c")
plt.savefig("../../figures/Chapter2/Problem2_10c", dpi=300)



# Part D
#################################################################################
x0 = y0 = 0
v0 = 1000
angles = (30,35,40, 45,50,55, 60)
distance_to_cliff = 10000
elevation_of_cliff = 10000


def bisection(f: callable, a: float, c: float) -> float:
    # Global Definitions
    EPS = 2.22044604925e-16
    ZERO = 1e-14
    MAX_ITERATIONS = 10000
    tolerance = 0.5 * EPS * (abs(a) + abs(c))
    f_a = f(a)
    f_c = f(c)
    if f_a * f_c > 0.0:
        raise NotImplementedError("Bounds don't seem to be surrounding a root")
        return None

    for iteration in range(MAX_ITERATIONS):
        b = 0.5 * (a + c)
        f_b = f(b)
        if (abs(c - a) < tolerance) or (abs(f_b) < ZERO):
            return b
        if f_b * f_a <= 0.0:
            c = b
        else:
            a = b
        f_a = f(a)
    raise UserWarning("Maximum number of iterations reached.")
    print("b = ",b)
    return b


def Final_pos(velocity):
    ic = np.array([x0, y0, velocity*np.cos(np.pi/3), velocity*np.sin(np.pi/4)])
    sim = ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate)
    sim.run()
    plt.plot(sim.X_series[0],sim.X_series[1])
    return sim.X_series[0][-1],sim.X_series[1][-1]


def F_velocity_x(velocity):
    ic = np.array([x0, y0, velocity*np.cos(np.pi/3), velocity*np.sin(np.pi/4)])
    sim = ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate)
    sim.run()
    return sim.X_series[0][-1]-distance_to_cliff


def F_velocity_y(velocity):
    ic = np.array([x0, y0, velocity*np.cos(np.pi/3), velocity*np.sin(np.pi/4)])
    sim = ODE(rhs, ic, ti=0, dt=0.01, tf=400, terminate=terminate)
    sim.run()
    return sim.X_series[1][-1]-elevation_of_cliff

print(bisection(F_velocity_x,600,1000))
print(bisection(F_velocity_y,600,1000))
plt.clf()
fig, ax = plt.subplots(1, 1)
#print(Final_pos(600))
#print(Final_pos(700))
#print(Final_pos(800))
print(Final_pos(833.7190674505091))
#print(Final_pos(900))
#print(Final_pos(1000))

if elevation_of_cliff<0:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(distance_to_cliff-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    valley = mpatches.Rectangle((distance_to_cliff,plt.ylim()[0]),(plt.xlim()[1]-distance_to_cliff),
                                (elevation_of_cliff-plt.ylim()[0]),color='green')
    ax.add_patch(ground)
    ax.add_patch(valley)
else:
    ground = mpatches.Rectangle((plt.xlim()[0],plt.ylim()[0]),(plt.xlim()[1]-plt.xlim()[0]),(0-plt.ylim()[0]),
                                color='green')
    ax.add_patch(ground)
    cliff = mpatches.Rectangle((distance_to_cliff, 0),(plt.xlim()[1]-distance_to_cliff),
                               elevation_of_cliff,color='green')
    ax.add_patch(cliff)

ax.grid()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")


plt.show()