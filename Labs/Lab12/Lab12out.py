#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1><b>Lab 12</b></h1>
# <h1>PHYS 580 - Computational Physics</h1>
# <h2>Professor Molnar</h2>
# </br>
# <h3><b>Ethan Knox</b></h3>
# <h4>https://www.github.com/ethank5149</h4>
# <h4>ethank5149@gmail.com</h4>
# </br>
# </br>
# <h3><b>November 19, 2020</b></h3>
# <hr>
# </center>

# In[1]:


import warnings
from pylj import md, mc, sample, util
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML, display

import ipywidgets as widgets
from ipywidgets import IntSlider as islider
from ipywidgets import FloatSlider as fslider

# plt.style.use('dark_background')
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'notebook')


# # Problem 1
# Use the starter programs (or your own equivalent ones) to simulate a system of, say, 25 particles in a square box of side length 5 (in units of $\sigma$, the Lennard-Jones parameter). Initially, let the particles be at rest but with positions shifted by relatively small random amounts away from the evenly spaced, square lattice vertices. (Why is it good to include such position variations?) Then, as the simulation proceeds, produce images of the time evolution of particle positions similar to those displayed in Fig. 9.6 of the textbook. In addition, reproduce the time series of the total energy, temperature, tagged particle and tagged pair separations. Are the fluctuations in energy and temperature, and the trend of the pair separation as you expect, and why?

# In[2]:


@widgets.interact(
    number_of_particles = islider(value=16,     min=1,   max=100,   step=1,   continuous_update=False, description='Particles'),
    box_length          = islider(value=150,    min=1,   max=1000,  step=1,  continuous_update=False, description='Box Length'),
    number_of_steps     = islider(value=10000,    min=1,   max=10000, step=1, continuous_update=False, description='Steps'),
    steps_per_frame     = islider(value=100,    min=1,   max=10000, step=1, continuous_update=False, description='Frequency'),
    temperature         = fslider(value=50, min=.15, max=473.15, step=1., continuous_update=False, description='Temperature'))
def md_simulation(number_of_particles, box_length, number_of_steps, steps_per_frame, temperature):
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Phase(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
    plt.savefig("Problem1.png")
    return system


# # Problem 2
# Find a way to speed up the convergence to equilibrium you observed in (1). In particular, use the feature of the starter program that allows one to change the kinetic energy of the particles via keyboard   input   during   the   simulation.   Similarly,  when   you   have   attained   a   stable   triangular arrangement of the particles (solid), find a way to melt it by heating the system. Demonstrate thatyou succeeded in melting the crystal by making appropriate plots of the particle arrangements and the time series of various functions. Note: discuss in your writeup what happens, and why, if the time step is too large (or too small) or if you raise the temperature by too much.

# In[3]:


@widgets.interact(
    number_of_particles = islider(value=9,     min=1,   max=27,   step=1,   continuous_update=False, description='Particles'),
    box_length          = islider(value=27,    min=3,   max=27,  step=1,  continuous_update=False, description='Box Length'),
    number_of_steps     = islider(value=10000,    min=1,   max=10000, step=1, continuous_update=False, description='Steps'),
    steps_per_frame     = islider(value=100,    min=1,   max=10000, step=1, continuous_update=False, description='Frequency'),
    temperature         = fslider(value=10.15, min=.15, max=473.15, step=1., continuous_update=False, description='Temperature'))
def md_simulation(number_of_particles, box_length, number_of_steps, steps_per_frame, temperature):
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Phase(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
    
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath((1.5 * i / number_of_steps + 1)*temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
    plt.savefig("Problem2a.png")
    return system


# For very low temperatures, if the particles begin stationary, nothing will evolve. This corresponds to zero temperature, and is regarded as impossible to reach. 
# 
# As you can see, once the system started to be heated halfway through the simulation (t = 1.0e-10), Not only did the energy noticably rise, but the particles began breaking apart from one another, just as desired.

# # Problem 3
# Study   how   varying   the   density,   initial   velocities   and/or   positions   affects   the   approach   to equilibrium   and   the   nature   of   the   final   equilibrium   configuration.   You   do   not   need   to   be exhaustive on this. For example, try putting 25 particles in a square of side length 10, and see how their characteristics change as you vary the temperature, substantiating your discussion with various time series graphs.

# In[4]:


@widgets.interact(
    number_of_particles = islider(value=25,     min=1,   max=100,   step=1,   continuous_update=False, description='Particles'),
    box_length          = islider(value=26,    min=1,   max=1000,  step=1,  continuous_update=False, description='Box Length'),
    number_of_steps     = islider(value=10000,    min=1,   max=10000, step=1, continuous_update=False, description='Steps'),
    steps_per_frame     = islider(value=100,    min=1,   max=10000, step=1, continuous_update=False, description='Frequency'),
    temperature         = fslider(value=25.15, min=.15, max=473.15, step=1., continuous_update=False, description='Temperature'))
def md_simulation(number_of_particles, box_length, number_of_steps, steps_per_frame, temperature):
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Phase(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
    plt.savefig("Problem3a.png")
    return system


# In[5]:


@widgets.interact(
    number_of_particles = islider(value=25,     min=1,   max=100,   step=1,   continuous_update=False, description='Particles'),
    box_length          = islider(value=100,    min=1,   max=1000,  step=1,  continuous_update=False, description='Box Length'),
    number_of_steps     = islider(value=10000,    min=1,   max=10000, step=1, continuous_update=False, description='Steps'),
    steps_per_frame     = islider(value=100,    min=1,   max=10000, step=1, continuous_update=False, description='Frequency'),
    temperature         = fslider(value=25.15, min=.15, max=473.15, step=1., continuous_update=False, description='Temperature'))
def md_simulation(number_of_particles, box_length, number_of_steps, steps_per_frame, temperature):
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Phase(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
    plt.savefig("Problem3b.png")
    return system


# In[6]:


@widgets.interact(
    number_of_particles = islider(value=25,     min=1,   max=100,   step=1,   continuous_update=False, description='Particles'),
    box_length          = islider(value=250,    min=1,   max=1000,  step=1,  continuous_update=False, description='Box Length'),
    number_of_steps     = islider(value=10000,    min=1,   max=10000, step=1, continuous_update=False, description='Steps'),
    steps_per_frame     = islider(value=100,    min=1,   max=10000, step=1, continuous_update=False, description='Frequency'),
    temperature         = fslider(value=25.15, min=.15, max=473.15, step=1., continuous_update=False, description='Temperature'))
def md_simulation(number_of_particles, box_length, number_of_steps, steps_per_frame, temperature):
    system = md.initialise(number_of_particles, temperature, box_length, 'square')
    sample_system = sample.Phase(system)
    system.time = 0
    for i in range(0, number_of_steps):
        system.integrate(md.velocity_verlet)
        system.md_sample()
        system.heat_bath(temperature)
        system.time += system.timestep_length
        system.step += 1
        if system.step % steps_per_frame == 0:
            sample_system.update(system)
            
    plt.savefig("Problem3c.png")
    return system


# In[ ]:




