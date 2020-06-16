########################################################################################################################
#     ========     |  Purdue Physics 580 - Computational Physics                                                       #
#     \\           |  Chapter 3 - Problem 17                                                                           #
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
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("Loading Data...")
x, dx = np.genfromtxt("C:/Users/ethan/CLionProjects/Purdue-PHYS-580/data/Problem3_17.csv",delimiter=',',
                      skip_header=1, unpack=True)
x_zoom, dx_zoom = x[np.where(x>2)], dx[np.where(x>2)]
print("Done, Plotting Full Plot...")

fig, ax = plt.subplots(1, 1, figsize=(11,8.5))
ax.scatter(x,dx,s=0.025)
ax.grid()
ax.set_xlabel(r"$\theta\,\,[rad]$")
ax.set_ylabel(r"$\dot{\theta}\,\,[rad/s]$")
ax.set_xlim(left=-3.2, right=3.2)
plt.suptitle(r"Problem 3.17 - Poincare Plot")
plt.savefig("../../figures/Chapter3/Problem3_17_full",dpi=2000)
# plt.savefig("../../figures/Chapter3/Problem3_17_full.pdf")
print("Done, Plotting Zoomed Plot...")

fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.scatter(x_zoom,dx_zoom,s=0.01)
ax.grid()
ax.set_xlabel(r"$\theta\,\,[rad]$")
ax.set_ylabel(r"$\dot{\theta}\,\,[rad/s]$")
ax.set_xlim(left=2,right=3.15)
ax.set_ylim(bottom=-1.3,top=-0.35)
plt.suptitle(r"Problem 3.17 - Poincare Plot")
plt.savefig("../../figures/Chapter3/Problem3_17", dpi=2000)
plt.savefig("../../figures/Chapter3/Problem3_17.pdf")
print("Done, Plotting 3D Parametric Plot...")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z = dx
x = np.sin(x)
y = np.cos(x)
ax.scatter(x, y, z,s=0.025)
ax.set_xlabel(r'$X$')
ax.set_ylabel(r'$Y$')
ax.set_zlabel(r'$Z$')
plt.suptitle("Problem 3.17 - Embedded Poincare Plot")
ax.view_init(45, 45)
plt.savefig("../../figures/Chapter3/Problem3_17_3D", dpi=2000)
# plt.savefig("../../figures/Chapter3/Problem3_17_3D.pdf")
print("Done, Plotting Multi-view 3D Parametric Plot...")


fig = plt.figure()
z = dx
x = np.sin(x)
y = np.cos(x)

ax1 = fig.add_subplot(2,2,1, projection='3d')
ax2 = fig.add_subplot(2,2,2, projection='3d')
ax3 = fig.add_subplot(2,2,3, projection='3d')
ax4 = fig.add_subplot(2,2,4, projection='3d')

ax1.view_init(45, 0)
ax2.view_init(45, 90)
ax3.view_init(45, 180)
ax4.view_init(45, 270)

ax1.scatter(x, y, z, s=0.02)
ax1.set_xlabel(r'$X$')
ax1.set_ylabel(r'$Y$')

ax2.scatter(x, y, z, s=0.02)
ax2.set_xlabel(r'$X$')
ax2.set_ylabel(r'$Y$')

ax3.scatter(x, y, z, s=0.02)
ax3.set_xlabel(r'$X$')
ax3.set_ylabel(r'$Y$')

ax4.scatter(x, y, z, s=0.02)
ax4.set_xlabel(r'$X$')
ax4.set_ylabel(r'$Y$')

plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.suptitle("Problem 3.17 - Embedded Poincare Plot")
plt.savefig("../../figures/Chapter3/Problem3_17_multi3D", dpi=2000)
# plt.savefig("../../figures/Chapter3/Problem3_17_multi3D.pdf")
print("Done")