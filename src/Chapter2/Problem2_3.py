########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 3                                                                            #
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

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]


def rhs(t, X, m,P,C,rho,eta,A,h):
    return np.array([P/(m*X[0])-eta*A*X[0]/(m*h)-0.5*C*rho*A*X[0]**2/m,])

def rhs2(t, X, m,P,C,rho,eta,A,h):
    return np.array([X[1],P/(m*X[1])-eta*A*X[1]/(h*m)-0.5*C*rho*A*X[1]**2/m])

P = 400
m = 70
v0 = 4
C = 0.5

w = 0.411  # Shoulder Width
A_front = 0.33
A_middle = 0.3*A_front
h_front = A_front/w
h_middle = A_middle/w
rho_air = 1.225
rho_water = 1000
eta_air = 2.0e-5
eta_water = 1.0e-3

sim1 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_air,eta=eta_air,A=A_front,h=h_front), np.array([v0,]), ti=0, dt=0.01, tf=100)
sim2 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_air,eta=eta_air,A=A_middle,h=h_middle), np.array([v0,]), ti=0, dt=0.01, tf=100)
sim3 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_water,eta=eta_water,A=A_front,h=h_front), np.array([v0,]), ti=0, dt=0.01, tf=10)
sim4 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_water,eta=eta_water,A=A_middle,h=h_middle), np.array([v0,]), ti=0, dt=0.01, tf=10)

sim1.run()
sim2.run()
sim3.run()
sim4.run()

# Plotting
fig, (ax1,ax2) = plt.subplots(1, 2)

ax1.plot(sim1.t_series, sim1.X_series[:,0], label=f"Front, Air")
ax1.plot(sim2.t_series, sim2.X_series[:,0], label=f"Middle, Air")
ax2.plot(sim3.t_series, sim3.X_series[:,0], label=f"Front, Water")
ax2.plot(sim4.t_series, sim4.X_series[:,0], label=f"Middle, Water")

ax1.legend()
ax1.grid()
ax1.set_xlabel("t")
ax1.set_ylabel(r"$v(t)$")

ax2.legend()
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel(r"$v(t)$")

plt.suptitle("Problem 2.3")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_3",dpi=300)