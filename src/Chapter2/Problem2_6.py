########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 2 - Problem 6                                                                            #
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
import numpy as np
from matplotlib import pyplot as plt

# Global Definitions
g = 9.81  # Gravitational Acceleration [m/s^2]
x_initial = y_initial = 0
v_initial = 700


def rhs(t, X):
    return np.array([X[2], X[3], 0, -g])


def terminate(X):
    return X[1] < 0


def f_ic(theta):
    return [x_initial, y_initial, v_initial*np.cos(np.pi*theta/180), v_initial*np.sin(np.pi*theta/180)]


def x_exact(t, x_initial, dx0):
    return x_initial+dx0*t


def y_exact(t, y_initial, dy0):
    return y_initial+dy0*t-0.5*g*t**2


angles = (30, 35, 40, 45, 50, 55)
t = np.linspace(0,200,20000)  # dt = 0.01

# Intentionally not using list comprehension for readability for non-python folks
y0 = euler(rhs, f_ic(angles[0]),t ,terminate=terminate)
y1 = euler(rhs, f_ic(angles[1]),t ,terminate=terminate)
y2 = euler(rhs, f_ic(angles[2]),t ,terminate=terminate)
y3 = euler(rhs, f_ic(angles[3]),t ,terminate=terminate)
y4 = euler(rhs, f_ic(angles[4]),t ,terminate=terminate)
y5 = euler(rhs, f_ic(angles[5]),t ,terminate=terminate)

y0_exact = [x_exact(t,y0[0,0],y0[2,0])[:np.size(y0[0])], y_exact(t,y0[1,0],y0[3,0])[:np.size(y0[1])]]
y1_exact = [x_exact(t,y1[0,0],y1[2,0])[:np.size(y1[0])], y_exact(t,y1[1,0],y1[3,0])[:np.size(y1[1])]]
y2_exact = [x_exact(t,y2[0,0],y2[2,0])[:np.size(y2[0])], y_exact(t,y2[1,0],y2[3,0])[:np.size(y2[1])]]
y3_exact = [x_exact(t,y3[0,0],y3[2,0])[:np.size(y3[0])], y_exact(t,y3[1,0],y3[3,0])[:np.size(y3[1])]]
y4_exact = [x_exact(t,y4[0,0],y4[2,0])[:np.size(y4[0])], y_exact(t,y4[1,0],y4[3,0])[:np.size(y4[1])]]
y5_exact = [x_exact(t,y5[0,0],y5[2,0])[:np.size(y5[0])], y_exact(t,y5[1,0],y5[3,0])[:np.size(y5[1])]]

# Adjust for the fact that the solutions could have been terminated early
y0_exact = [y0_exact[0][:np.size(y0[0])], y0_exact[1][:np.size(y0[1])]]
y1_exact = [y1_exact[0][:np.size(y1[0])], y1_exact[1][:np.size(y1[1])]]
y2_exact = [y2_exact[0][:np.size(y2[0])], y2_exact[1][:np.size(y2[1])]]
y3_exact = [y3_exact[0][:np.size(y3[0])], y3_exact[1][:np.size(y3[1])]]
y4_exact = [y4_exact[0][:np.size(y4[0])], y4_exact[1][:np.size(y4[1])]]
y5_exact = [y5_exact[0][:np.size(y5[0])], y5_exact[1][:np.size(y5[1])]]
t0 = t[:np.size(y0[0])]
t1 = t[:np.size(y1[0])]
t2 = t[:np.size(y2[0])]
t3 = t[:np.size(y3[0])]
t4 = t[:np.size(y4[0])]
t5 = t[:np.size(y5[0])]



# Plotting
# First Plot
fig, ax = plt.subplots(1, 1)
ax.plot(y0[0]/1000, y0[1]/1000, label=rf"$\theta = {angles[0]}^{{\circ}}$")
ax.plot(y1[0]/1000, y1[1]/1000, label=rf"$\theta = {angles[1]}^{{\circ}}$")
ax.plot(y2[0]/1000, y2[1]/1000, label=rf"$\theta = {angles[2]}^{{\circ}}$")
ax.plot(y3[0]/1000, y3[1]/1000, label=rf"$\theta = {angles[3]}^{{\circ}}$")
ax.plot(y4[0]/1000, y4[1]/1000, label=rf"$\theta = {angles[4]}^{{\circ}}$")
ax.plot(y5[0]/1000, y5[1]/1000, label=rf"$\theta = {angles[5]}^{{\circ}}$")
ax.legend()
ax.grid()
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
plt.suptitle("Problem 2.6a")
plt.savefig("../../figures/Chapter2/Problem2_6a", dpi=300)

# Second Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t0, np.absolute(y0[0]-y0_exact[0]), label=rf"$\theta = {angles[0]}^{{\circ}}$")
ax1.plot(t1, np.absolute(y1[0]-y1_exact[0]), label=rf"$\theta = {angles[1]}^{{\circ}}$")
ax1.plot(t2, np.absolute(y2[0]-y2_exact[0]), label=rf"$\theta = {angles[2]}^{{\circ}}$")
ax1.plot(t3, np.absolute(y3[0]-y3_exact[0]), label=rf"$\theta = {angles[3]}^{{\circ}}$")
ax1.plot(t4, np.absolute(y4[0]-y4_exact[0]), label=rf"$\theta = {angles[4]}^{{\circ}}$")
ax1.plot(t5, np.absolute(y5[0]-y5_exact[0]), label=rf"$\theta = {angles[5]}^{{\circ}}$")

ax2.plot(t0, np.absolute(y0[1]-y0_exact[1]), label=rf"$\theta = {angles[0]}^{{\circ}}$")
ax2.plot(t1, np.absolute(y1[1]-y1_exact[1]), label=rf"$\theta = {angles[1]}^{{\circ}}$")
ax2.plot(t2, np.absolute(y2[1]-y2_exact[1]), label=rf"$\theta = {angles[2]}^{{\circ}}$")
ax2.plot(t3, np.absolute(y3[1]-y3_exact[1]), label=rf"$\theta = {angles[3]}^{{\circ}}$")
ax2.plot(t4, np.absolute(y4[1]-y4_exact[1]), label=rf"$\theta = {angles[4]}^{{\circ}}$")
ax2.plot(t5, np.absolute(y5[1]-y5_exact[1]), label=rf"$\theta = {angles[5]}^{{\circ}}$")

ax1.legend()
ax1.grid()
ax1.set_ylabel("x [m]")
ax1.set_title("Absolute Error")

ax2.legend()
ax2.grid()
ax2.set_xlabel("t [s]")
ax2.set_ylabel("y [m]")

plt.suptitle("Problem 2.6b")
plt.savefig("../../figures/Chapter2/Problem2_6b", dpi=300)
