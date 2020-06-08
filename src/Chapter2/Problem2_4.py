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
from lib.NDSolveSystem import ODE, SymplecticODE
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
w = 0.411  # Shoulder width


def rhs(t, X, m,P,C,rho,eta,A,h,theta):
    return np.array([P/(m*X[0])-g*np.sin(theta)-eta*w*X[0]/m-0.5*C*rho*A*X[0]**2/m,])

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
theta = np.pi*10/180

sim1 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_air,eta=eta_air,A=A_front,h=h_front,theta=0), np.array([v0,]), ti=0, dt=0.01, tf=100)
sim2 = ODE(partial(rhs, m=m,P=P,C=C,rho=rho_air,eta=eta_air,A=A_front,h=h_front,theta=-theta), np.array([v0,]), ti=0, dt=0.01, tf=100)

sim1.run()
sim2.run()

# Plotting
fig, ax = plt.subplots(1, 1)

ax.plot(sim1.t_series, sim1.X_series[:,0], label=f"0 deg")
ax.plot(sim2.t_series, sim2.X_series[:,0], label=f"-10 deg")

ax.legend()
ax.grid()
ax.set_xlabel("t")
ax.set_ylabel(r"$v(t)$")


plt.suptitle("Problem 2.4")
plt.subplots_adjust(wspace=0.3)
plt.savefig("../../figures/Chapter2/Problem2_4",dpi=300)
