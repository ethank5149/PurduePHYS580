########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 2                                                                            #
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
from lib import Constants
import numpy as np
from matplotlib import pyplot as plt
from functools import partial

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
air = Constants.Air()


def rhs(t, X, m,P,C,rho,A):
    return np.array([P/(m*X[0])-0.5*C*rho*A*X[0]**2/m,])

def rhs2(t, X, m,P,C,rho,A):
    return np.array([X[1],P/(m*X[1])-0.5*C*rho*A*X[1]**2/m])

def W_d(X,m,P,C,rho,A):
    return X[:,0]*0.5*C*rho*A*X[:,1]**2

def v(t,m,P,v0):
    return np.sqrt(2*P*t/m+v0**2)


P = 400
m = 70
v0 = 4
C = 0.5
A = 0.33

curried_rhs_front = partial(rhs, m=m,P=P,C=C,rho=air.density,A=A)
curried_rhs_middle = partial(rhs, m=m,P=P,C=C,rho=air.density,A=0.3*A)
curried_rhs2_front = partial(rhs2, m=m,P=P,C=C,rho=air.density,A=A)
curried_rhs2_middle = partial(rhs2, m=m,P=P,C=C,rho=air.density,A=0.3*A)

ic = np.array([v0,])
ic2 = np.array([0,v0])

sim1 = ODE(curried_rhs_front,ic, ti=0, dt=0.01, tf=200)
sim2 = ODE(curried_rhs_middle, ic, ti=0, dt=0.01, tf=200)
sim3 = ODE(curried_rhs2_front, ic2, ti=0, dt=0.01, tf=200)
sim4 = ODE(curried_rhs2_middle, ic2, ti=0, dt=0.01, tf=200)

sim1.run()
sim2.run()
sim3.run()
sim4.run()

# Plotting
fig, ax = plt.subplots(2, 1,sharex=True)

ax[0].plot(sim1.t_series, sim1.X_series[:,0], label=f"Front of the pack")
ax[0].plot(sim2.t_series, sim2.X_series[:,0], label=f"Middle of the pack")
ax[1].plot(sim3.t_series, W_d(sim3.X_series,m,P,C,air.density,A)-W_d(sim4.X_series,m,P,C,air.density,0.3*A), label=r"$W_d$ Front - $W_d$ Middle")
# ax[1].plot(sim4.t_series, W_d(sim4.X_series,m,P,C,rho,0.3*A), label=f"Middle of the pack")

for a in ax:
    a.legend()
    a.grid()
    a.set_xlabel("t")

ax[0].set_ylabel(r"$v(t)$")
ax[1].set_ylabel(r"$W [J]$")

plt.suptitle("Problem 2.2")
plt.savefig("../../figures/Chapter2/Problem2_2",dpi=300)
