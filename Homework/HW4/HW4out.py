#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1><b>Homework 4</b></h1>
# <h1>PHYS 580 - Computational Physics</h1>
# <h2>Professor Molnar</h2>
# </br>
# <h3><b>Ethan Knox</b></h3>
# <h4>https://www.github.com/ethank5149</h4>
# <h4>ethank5149@gmail.com</h4>
# </br>
# </br>
# <h3><b>October 30, 2020</b></h3>
# </center>
# <hr>

# # Problem 1
# ## Problem 7.2 (p.188)
# 
# Simulate a random walk in three dimensions allowing the walker to make steps of unit length in random directions; don't restrict the walker to sites on a discrete lattice. Show that the motion is diffusive, that is, $\left<r^2\right>\sim t$. Find the value of the proportionality constant.

# In[2]:


import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm, trange
from scipy.spatial.distance import cdist
from IPython.display import display, Math


np.set_printoptions(sign=' ', linewidth=100, precision=4, suppress=True)
plt.style.use('dark_background')
rng = default_rng()


# In[4]:


def RW(num_steps=1000000):
    theta = 2 * np.pi * rng.random(num_steps)
    phi = np.pi * rng.random(num_steps)   
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    steps = np.vstack((x, y, z))
    positions = np.cumsum(steps, 1)
    return positions

def RW_diffusion(num_steps=1000, num_walkers=1000, func=lambda _ : _):
    rng = np.random.default_rng()
    ensemble = np.zeros((num_walkers, num_steps))
    for i in range(num_walkers):
        theta = 2 * np.pi * rng.random(num_steps)
        phi = np.pi * rng.random(num_steps)   
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        steps = np.vstack((x, y, z))
        positions = np.cumsum(steps, 1)
        ensemble[i,:] = func(np.power(np.apply_along_axis(np.linalg.norm, 0, positions), 2))
    return ensemble


# ## 3D Random Walk
# ### Path | $\left(x_n,y_n,z_n\right)$

# In[6]:


rw = RW()


# In[8]:


fig = plt.figure(figsize=(16,16), dpi=200)
ax = fig.gca(projection="3d")
ax.set_title("Random Walk")
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.plot(rw[0], rw[1], rw[2], lw=1)
plt.savefig('plots/Problem_1a.png')


# ### Mean Radial Distance (Squared) | $\left<r^2_n\right>$

# In[10]:


ensemble = RW_diffusion()
mean_distances = np.apply_along_axis(np.mean, 0, ensemble)


# In[12]:


fit_func = lambda x, a, b : a + b * x
fit_x = range(mean_distances.size)
(a, b), _ = curve_fit(fit_func, fit_x, mean_distances)


# In[14]:


fig, ax = plt.subplots(1,1,figsize=(16,16), dpi=200)
ax.plot(mean_distances)
ax.plot(fit_x, fit_func(fit_x, a, b), ls='--', label=fr'$y\sim{a:0.4f}+{b:0.4f}x$')
ax.set_title(r'3D Random Walk | Mean Radial Distance (Squared) | $\left<r^2_n\right>$')
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$\left<r^2\right>$')
ax.legend()
ax.grid()
plt.savefig('plots/Problem_1b.png')


# # Problem 2
# ## 7.6 (p.194)
# 
# Simulate SAWs in three dimensions. Determine the variation of $\left<r^2\right>$ with step number and find the value of $\nu$, where this parameter is defined through the relation (7.9). Compare your results with those in Figure 7.6. You should find that $\nu$ decreases for successively higher dimensions. (It is $1$ in one dimension and $3/4$ in two dimensions.) Can you explain this trend qualitatively?
# 
# Also check whether you can reproduce the analytic result $ν = 3/4$ for SAW on a 2D grid.

# In[16]:


rotation_matrices = np.array([
            [[  1,  0,  0], [  0,  0, -1], [  0,  1,  0]],
            [[  1,  0,  0], [  0, -1,  0], [  0,  0, -1]],
            [[  1,  0,  0], [  0,  0,  1], [  0, -1,  0]],
            [[  0,  0,  1], [  0,  1,  0], [ -1,  0,  0]],
            [[ -1,  0,  0], [  0,  1,  0], [  0,  0, -1]],
            [[  0,  0, -1], [  0,  1,  0], [ -1,  0,  0]],
            [[  0, -1,  0], [  1,  0,  0], [  0,  0,  1]],
            [[ -1,  0,  0], [  0, -1,  0], [  0,  0,  1]],
            [[  0,  1,  0], [ -1,  0,  0], [  0,  0,  1]]])

def SAW(num_steps=1000, steps=1000):
    init_state = np.dstack((np.arange(num_steps),np.zeros(num_steps),np.zeros(num_steps)))[0]
    state = init_state.copy()
    acpt = 0
    while acpt <= steps:
        pick_pivot = np.random.randint(1, num_steps - 1)
        pick_side = np.random.choice([-1, 1])
        if pick_side == 1:
            old_chain = state[0 : pick_pivot + 1]
            temp_chain = state[pick_pivot + 1 : ]
        else:
            old_chain = state[pick_pivot : ]
            temp_chain = state[0 : pick_pivot]
        symtry_oprtr = rotation_matrices[np.random.randint(len(rotation_matrices))]
        new_chain = np.apply_along_axis(lambda _: np.dot(symtry_oprtr, _), 1, temp_chain - state[pick_pivot]) + state[pick_pivot]
        overlap = cdist(new_chain,old_chain)
        overlap = overlap.flatten()
        if len(np.nonzero(overlap)[0]) != len(overlap):
            continue
        else:
            if pick_side == 1:
                state = np.concatenate((old_chain, new_chain), axis=0)
            elif pick_side == -1:
                state = np.concatenate((new_chain, old_chain), axis=0)
            acpt += 1
    return state - np.int_(state[0])


def SAW_diffusion(num_steps=1000, num_walkers=1000, steps=100, func=lambda _ : _):
    rng = np.random.default_rng()
    ensemble = np.zeros((num_walkers, num_steps))
    for i in trange(num_walkers):
        init_state = np.dstack((np.arange(num_steps),np.zeros(num_steps),np.zeros(num_steps)))[0]
        state = init_state.copy()
        acpt = 0
        while acpt <= steps:
            pick_pivot = np.random.randint(1, num_steps - 1)
            pick_side = np.random.choice([-1, 1])
            if pick_side == 1:
                old_chain = state[0 : pick_pivot + 1]
                temp_chain = state[pick_pivot + 1 : ]
            else:
                old_chain = state[pick_pivot : ]
                temp_chain = state[0 : pick_pivot]
            symtry_oprtr = rotation_matrices[np.random.randint(len(rotation_matrices))]
            new_chain = np.apply_along_axis(lambda _: np.dot(symtry_oprtr, _), 1, temp_chain - state[pick_pivot]) + state[pick_pivot]
            overlap = cdist(new_chain,old_chain)
            overlap = overlap.flatten()
            if len(np.nonzero(overlap)[0]) != len(overlap):
                continue
            else:
                if pick_side == 1:
                    state = np.concatenate((old_chain, new_chain), axis=0)
                elif pick_side == -1:
                    state = np.concatenate((new_chain, old_chain), axis=0)
                acpt += 1
        ensemble[i,:] = func(np.power(np.apply_along_axis(np.linalg.norm, 1, state - np.int_(state[0])), 2))
    return ensemble


# ## 3D Self-Avoiding Random Walk
# ### Path | $\left(x_n,y_n,z_n\right)$

# In[18]:


saw = SAW()


# In[20]:


fig = plt.figure(figsize=(16,16), dpi=200)
ax = fig.gca(projection="3d")
ax.set_title(r"3D Self-Avoiding Random Walk | Path | $\left(x_n,y_n,z_n\right)$")
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.plot(saw[:,0], saw[:,1], saw[:,2], lw=1)
plt.savefig('plots/Problem_2a.png')


# ### Mean Radial Distance (Squared) | $\left<r^2_n\right>$

# In[22]:


ensemble = SAW_diffusion()
mean_distances = np.apply_along_axis(np.mean, 0, ensemble)


# In[24]:


fit_func = lambda x, c, v : c * np.power(x, 2 * v)
fit_x = range(mean_distances.size)
(c, v), _ = curve_fit(fit_func, fit_x, mean_distances)
display(Math(fr'\text{{Flory Exponent }}(\nu)\approx{v:0.6f}')) 


# In[26]:


fig, ax = plt.subplots(1,1,figsize=(12,12), dpi=200)
ax.plot(mean_distances)
ax.plot(fit_x, fit_func(fit_x, c, v), ls='--', label=fr'$y\sim{c:0.4f}x^{{2\cdot{v:0.4f}}}$')
ax.set_title(r'3D Self-Avoiding Random Walk | Mean Radial Distance (Squared) | $\left<r^2_n\right>$')
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'$\left<r^2\right>$')
ax.legend()
ax.grid()
plt.savefig('plots/Problem_2b.png')


# For the 2D case, see lab 9 (I was able to replicate $\nu\approx0.75$)

# # Problem 3
# ## 7.12 (p.205)
# 
# Calculate the entropy for the cream-in-your-coffee problem, and reproduce the results in Figure 7.16.
# 
# If you have trouble with running time, then try first on coarser grids (e.g., $50\times50$ for the random walk, $4\times4$ for the entropy, and only $100$ particles)

# In[28]:


def RW(num_steps=100000, num_particles=1000):
    rng = np.random.default_rng()
    theta = 2 * np.pi * rng.random(num_steps * num_particles)
    r = rng.random(num_steps * num_particles)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    steps = np.vstack((x, y))    
    steps = steps.reshape((2, num_steps, num_particles))
    positions = np.cumsum(steps, 1)    
    return positions

def RW_dist(num_steps=100000, num_particles=1000):
    positions = RW(num_steps, num_particles)
    return np.apply_along_axis(np.linalg.norm, 0, positions) ** 2


# In[30]:


rw = RW(num_particles=12)


# In[31]:


fig, ax = plt.subplots(1,1,figsize=(16,16), dpi=200)
ax.set_title("Random Walk")
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
for i in range(rw[0,0,:].size):
    ax.plot(rw[0,:,i], rw[1,:,i], lw=1)
plt.savefig('plots/Problem_3a.png')


# In[32]:


rw = RW(num_particles=9)
rw_dist = RW_dist(num_particles=9)


# In[33]:


fig, ax = plt.subplots(1,1,figsize=(16,16), dpi=200)
ax.set_title("Random Walk | Distance")
ax.set_xlabel('Iteration')
ax.set_ylabel('Distance [m]')
for i in range(rw_dist.shape[1]):
    ax.plot(rw_dist[:,i], lw=1)
ax.plot(np.mean(rw_dist, axis=1), lw=1, label='Average')
ax.legend()
plt.savefig('plots/Problem_3b.png')


# # Problem 4
# ## 7.15 (p.205)
# 
# Perform the random-walk simulation of spreading cream (Figures 7.13 and 7.14), and let one of the walls of the container possess a small hole so that if a cream particle enters the hole, it leaves the container. Calculate the number of particles in the container as a function of time. Show that this number, which is proportional to the partial pressure of the cream particles varies as $e^{-\frac{t}{\tau}}$, where $\tau$ is the effective time constant for the escape. _Hint:_ Reasonable parameter choices are a $50\times50$ container lattice and a hole $10$ units in length along one of the edges.

# Omitted to preserve sanity
