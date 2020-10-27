#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1><b>Lab 8</b></h1>
# <h1>PHYS 580 - Computational Physics</h1>
# <h2>Professor Molnar</h2>
# </br>
# <h3><b>Ethan Knox</b></h3>
# <h4>https://www.github.com/ethank5149</h4>
# <h4>ethank5149@gmail.com</h4>
# </br>
# </br>
# <h3><b>October 22, 2020</b></h3>
# <hr>
# </center>

# In[98]:


import numpy as np
from numpy import pi, sin, cos, sqrt, apply_along_axis
from scipy.integrate import quad
from functools import partial
import matplotlib.pyplot as plt


# In[99]:


get_ipython().run_line_magic('run', 'paths.py')
get_ipython().run_line_magic('run', 'integrands.py')


# # Problem 1
# The starter code ```loop.py``` provided uses the simple rectangular panel method of integration to compute the magnetic field $\mathbf{B}$ of a current loop (the Matlab code has two files ```loop.m``` and ```loop_calculate_field.m```). The loop is in the $x-y$ plane and is centered at the origin, just like in the setup discussed in class. (By the way, for this problem, the rectangular panel method is actually the same as the trapezoidal rule. Why?) Extend the programs (or create your own equivalent ones) to calculate the $\mathbf{B}$ field of two identically shaped parallel loops that share a common axis (the z-axis) and are situated symmetrically about the origin with their respective planes a distance $d$ apart. First, calculate the field when the loops carry equal currents in the same direction, which is the Helmholtz coil configuration that is known to produce a nearly uniform magnetic field at the center. Then, investigate what happens if equal currents are carried in the opposite directions.

# In[100]:


l = 10
r = 10
n = 1
n_loose = 10
n_tight = 20
I = 1
n_quiver = 12


# In[101]:


x = np.linspace(-r, r, n_quiver)
y = np.linspace(-r, r, n_quiver)
z = np.linspace(-l, l, n_quiver)
points = np.vstack((x,y,z))
s = np.linspace(0, 1, 1000)
xx2d, zz2d = np.meshgrid(x, z)
xx, yy, zz = np.meshgrid(x, y, z)


# In[102]:


B_loop_top_func = partial(biot_savart,     path=partial(path_loop_top, r=r, l=l, n=n),     dpath=partial(dpath_loop_top, r=r, l=l, n=n), I=I)

B_loop_bottom_func = partial(biot_savart,     path=partial(path_loop_bottom, r=r, l=l, n=n),     dpath=partial(dpath_loop_bottom, r=r, l=l, n=n), I=I)

B_rloop_bottom_func = partial(biot_savart,     path=partial(path_rloop_bottom, r=r, l=l, n=n),     dpath=partial(dpath_rloop_bottom, r=r, l=l, n=n), I=I)

B_solenoid_tight_func = partial(biot_savart,     path=partial(path_solenoid, r=r, l=l, n=n_tight),     dpath=partial(dpath_solenoid, r=r, l=l, n=n_tight), I=I)

B_solenoid_loose_func = partial(biot_savart,     path=partial(path_solenoid, r=r, l=l, n=n_loose),     dpath=partial(dpath_solenoid, r=r, l=l, n=n_loose), I=I)


# In[103]:


B_loop_top       = apply_along_axis(B_loop_top_func, 0, points)
B_loop_bottom    = apply_along_axis(B_loop_bottom_func, 0, points)
B_rloop_bottom   = apply_along_axis(B_rloop_bottom_func, 0, points)
B_solenoid_tight = apply_along_axis(B_solenoid_tight_func, 0, points).T
B_solenoid_loose = apply_along_axis(B_solenoid_loose_func, 0, points).T


# In[104]:


B_loop_parallel = (B_loop_top + B_loop_bottom).T
B_loop_opposite = (B_loop_top - B_loop_bottom).T
B_loop_ropposite = (B_loop_top + B_rloop_bottom).T


# In[105]:


fig = plt.figure(figsize=(16, 16), dpi=200)
ax = fig.gca(projection='3d')
wire_a = ax.plot(*partial(path_loop_top, l=l, r=r, n=n)(s), color='red')
wire_b = ax.plot(*partial(path_loop_bottom, l=l, r=r, n=n)(s), color='red')
quiver = ax.quiver(xx, yy, zz, *np.hsplit(B_loop_parallel, 3), color='k', normalize=True)

ax.set_title(r'$\vec{B}$ | Parallel Current Loops')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(0, 90)
plt.savefig('Problem_1a.png')


# In[106]:


fig = plt.figure(figsize=(16, 16), dpi=200)
ax = fig.gca(projection='3d')
wire_a = ax.plot(*partial(path_loop_top, l=l, r=r, n=n)(s), color='red')
wire_b = ax.plot(*partial(path_loop_bottom, l=l, r=r, n=n)(s), color='blue')
quiver = ax.quiver(xx, yy, zz, *np.hsplit(B_loop_ropposite, 3), color='k', normalize=True)

ax.set_title(r'$\vec{B}$ | Opposite Current Loops')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(0, 90)
plt.savefig('Problem_1b.png')


# # Problem 2
# Extend your programs further to implement the same calculation for a helical coil of multiple loops (i.e., a current-carrying wire wrapped around a cylinder). The coil is centered at the origin, its axis coincides with the $z$ axis, and it has a given pitch $P$ (i.e., the location of the wire advances by distance $P$ in the axial direction as the angle of winding along the wire goes through $2\pi$). Using your program, calculate the magnetic field both inside and outside the coil for various lengths and numbers of winding, with the current set such that $\frac{\mu_0}{4\pi}=1$. Show results for at least one case with loose winding, illustrating how $\textbf{B}$ leaks through the coil, and a case with tight winding, showing how $\textbf{B}$ approaches the field of an ideal solenoid.
# 
# __Note:__ the analytic result for $\textbf{B}$ on the axis of an ideal (i.e., tightly wound) solenoid of radius $R$ extending along the $z$-axis from $z=-\frac{L}{2}$ to $\frac{L}{2}$ is:
# 
# $$\textbf{B}=\hat{e}_z\frac{\mu_0In}{2}\left(\frac{\frac{L}{2}+z}{\sqrt{\left(\frac{L}{2}+z\right)^2+R^2}}+\frac{\frac{L}{2}-z}{\sqrt{\left(\frac{L}{2}-z\right)^2+R^2}}\right)$$
# 
# where  $n$  is the winding number per unit length and the current  $I$  is counter-clockwise when viewed from above the $x$-$y$ plane. However, away from the axis, for example, at a point on the $x$ axis, $\textbf{B}$ is much harder to calculate or approximate analytically (especially when $x$ is of the same order as $R$).

# In[107]:


fig = plt.figure(figsize=(16,16), dpi=200)
ax = fig.gca(projection='3d')
solenoid = ax.plot(*partial(path_solenoid, l=l, r=r, n=n_tight)(s), color='red')
quiver = ax.quiver(xx, yy, zz, *np.hsplit(B_solenoid_tight, 3), color='k', normalize=True)

ax.set_title(r'$\vec{B}$ | Tightly Wound Solenoid')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(0, 90)
plt.savefig('Problem_2a.png')


# In[108]:


fig = plt.figure(figsize=(16,16), dpi=200)
ax = fig.gca(projection='3d')
solenoid = ax.plot(*partial(path_solenoid, l=l, r=r, n=n_loose)(s), color='red')
quiver = ax.quiver(xx, yy, zz, *np.hsplit(B_solenoid_loose, 3), color='k', normalize=True)

ax.set_title(r'$\vec{B}$ | Loosely Wound Solenoid')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(0, 90)
plt.savefig('Problem_2b.png')


# In[109]:


fig = plt.figure(figsize=(16,16), dpi=200)
ax = fig.gca(projection='3d')
quiver = ax.quiver(xx, yy, zz, *np.hsplit(B_solenoid_tight - B_solenoid_loose, 3), color='k', normalize=True)

ax.set_title(r'$\Delta\vec{B}$ | Tightly Wound - Loosely Wound')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.view_init(0, 90)
plt.savefig('Problem_2c.png')


# In[ ]:




