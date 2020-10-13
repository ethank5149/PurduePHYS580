#!/usr/bin/env python
# coding: utf-8

# <center>
# <h1><b>Lab 6</b></h1>
# <h1>PHYS 580 - Computational Physics</h1>
# <h2>Professor Molnar</h2>
# </br>
# <h3><b>Ethan Knox</b></h3>
# <h4>https://www.github.com/ethank5149</h4>
# <h4>ethank5149@gmail.com</h4>
# </br>
# </br>
# <h3><b>October 8, 2020</b></h3>
# <hr>
# </center>

# #### Imports

# In[1]:


from scipy.integrate import solve_ivp
from functools import partial
from numpy import apply_along_axis as thread
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from itertools import product

np.set_printoptions(sign=' ', linewidth=100, precision=4, suppress=True)
plt.style.use('dark_background')


# #### Ephemerides

# In[2]:


saturn_pos   = np.asarray([ 0.000000000000000E-00,  0.000000000000000E-00,  0.000000000000000E-00])
saturn_vel   = np.asarray([ 0.000000000000000E-00,  0.000000000000000E-00,  0.000000000000000E-00])
saturn_m     = 1.0
titan_pos    = np.asarray([-2.349442311566901E-03,  7.153850305740229E-03, -3.453819029042604E-03])
titan_vel    = np.asarray([-3.051170257091698E-03, -6.116575942904424E-04,  6.191676545191825E-04])
titan_m      = 2.367e-4
rhea_pos     = np.asarray([-1.255393996976257E-03,  2.964327117947999E-03, -1.436487271404112E-03])
rhea_vel     = np.asarray([-4.561058618370766E-03, -1.378619573613746E-03,  1.133396533665187E-03])
rhea_m       = 4.062e-6
hyperion_pos = np.asarray([ 3.643573903413987E-03, -7.406358781088398E-03,  3.362810684983125E-03])
hyperion_vel = np.asarray([ 2.818505976172301E-03,  1.278934691098328E-03, -9.114757023594637E-04])
hyperion_m   = 9.754e-9
hyperion_r   = 9.2653e-7
hyperion_T   = 21.28


# #### Tweakable Parameters

# In[3]:


theta, dtheta = np.pi, 2 * np.pi / hyperion_T
phi, dphi = np.pi, 2 * np.pi / hyperion_T
L = 2.0 * hyperion_r

duration = 22.0
fps = 24.0
G = 8.458e-8
k = 1.2e3
global_fig_size = (16,16)
alpha = 0.5


# #### Helper Functions

# In[4]:


def rhs(t, y, mu, k, L, G):
    m1, m2, m3, m4 = mu
    x1, x2, x3, x4, v1, v2, v3, v4 = np.hsplit(y, 8)
    return np.hstack((
        v1, v2, v3, v4,
        -G*(m2*(x1-x2)/norm(x1-x2)**3+m3*(x1-x3)/norm(x1-x3)**3+m4*(x1-x4)/norm(x1-x4)**3),
        -G*(m1*(x2-x1)/norm(x2-x1)**3+m3*(x2-x3)/norm(x2-x3)**3+m4*(x2-x4)/norm(x2-x4)**3),
        -G*(m1*(x3-x1)/norm(x3-x1)**3+m2*(x3-x2)/norm(x3-x2)**3+m4*(x3-x4)/norm(x3-x4)**3)+k*(norm(x4-x3)-L)**2*(x4-x3)/norm(x4-x3),
        -G*(m1*(x4-x1)/norm(x4-x1)**3+m2*(x4-x2)/norm(x4-x2)**3+m3*(x4-x3)/norm(x4-x3)**3)+k*(norm(x3-x4)-L)**2*(x3-x4)/norm(x3-x4)
    ))


# In[5]:


def COM(x2, x3, alpha):
    return (x2 + (alpha ** (-1) - 1) * x3) / (1 + (alpha ** (-1) - 1))


# In[6]:


def cart2sph(x, y, z):
    return np.sqrt(z**2+y**2+z**2), np.arctan2(y, x), np.arctan2(np.sqrt(x**2+y**2), z)

def sph2cart(r, theta, phi):
    return r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)

def dsph2dcart(r, dr, theta, dtheta, phi, dphi):
    return dr*np.sin(theta)*np.cos(phi)+r*(dtheta*np.cos(theta)*np.cos(phi)-dphi*np.sin(theta)*np.sin(phi)),
    dr*np.sin(theta)*np.sin(phi)+r*(dtheta*np.cos(theta)*np.sin(phi)+dphi*np.sin(theta)*np.cos(phi)),
    dr*np.cos(theta)-dtheta*r*np.sin(theta)


# In[7]:


def run_simulation(theta, dtheta, phi, dphi, fps, duration, k, G, alpha=0.5, vel_boost=1.0):
    hyperion_a_pos=hyperion_pos+np.asarray([*sph2cart(hyperion_r, theta, phi)])
    hyperion_a_vel=vel_boost*hyperion_vel+np.asarray([*dsph2dcart(hyperion_r, 0, theta, dtheta, phi, dphi)])
    hyperion_a_m=alpha*hyperion_m
    
    hyperion_b_pos=hyperion_pos-np.asarray([*sph2cart(hyperion_r, theta, phi)])
    hyperion_b_vel=vel_boost*hyperion_vel-np.asarray([*dsph2dcart(hyperion_r, 0, theta, dtheta, phi, dphi)])
    hyperion_b_m=(1-alpha)*hyperion_m

    mu = saturn_m, titan_m, hyperion_a_m, hyperion_b_m
    t_eval = np.linspace(0.0, duration, num=int(fps*duration))
    L = 2.0 * np.average(hyperion_r)
    df = partial(rhs, k=k, L=L, mu=mu, G=G)
    state0 = np.hstack((
        saturn_pos, titan_pos, hyperion_a_pos, hyperion_b_pos,
        saturn_vel, titan_vel, hyperion_a_vel, hyperion_b_vel
    ))
    sol = solve_ivp(df, [0.0, duration], state0, t_eval=t_eval, rtol=1.0e-3, atol=1.0e-5)
    if not sol.success:
        print(sol.message)
        raise RuntimeError
    return *np.vsplit(sol.y, 8), sol.t


# ## Problem 1
# 
# Use the provided starter program `lab6-hyperion.py` (or your own equivalent program) to study the motion of Hyperion, one of Saturn’s moons, in the dumbbell model discussed in class. 
# 
# - First, observe and display the orbital motion and the motion of the dumbbell axis. 
# - In particular, try different initial conditions for the Hyperion center of mass location and velocity, and the dumbbell axis orientation and angular velocity. 
# - Study both the case of a hypothetical circular orbit as well as a few (2-3) elliptic orbits with different eccentricities. 
# - Make sure to include the real orbit of Hyperion with perihelion $a(1-e)\approx1$, $3\times10^6$ km$=1$ HU (“Hyperion unit”), and eccentricity $e = 0.123$. 
# - How would you characterize the nature of the spinning motion of Hyperion, based on your simulations?  
# - Does the kind of spinning motion depend on the type of orbit, and if yes, how?

# In[8]:


x0, x1, x2, x3, v0, v1, v2, v3, t = run_simulation(theta, dtheta, phi, dphi, fps, duration, k, G)


# In[9]:


## Bounds
min_x, max_x = min(np.min(x2[0,:]),np.min(x3[0,:])), max(np.max(x2[0,:]),np.max(x3[0,:]))
min_y, max_y = min(np.min(x2[1,:]),np.min(x3[1,:])), max(np.max(x2[1,:]),np.max(x3[1,:]))
min_z, max_z = min(np.min(x2[2,:]),np.min(x3[2,:])), max(np.max(x2[2,:]),np.max(x3[2,:]))
## Center of Mass
com = COM(x2, x3, 0.5)


# In[10]:


fig = plt.figure(figsize=global_fig_size, dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Orbital Motion')
ax.plot(x0[0,:], x0[1,:], x0[2,:], label=r'$Saturn$')
ax.plot(x1[0,:], x1[1,:], x1[2,:], label=r'$Titan$')
ax.plot(x2[0,:], x2[1,:], x2[2,:], label=r'$Hyperion_a$')
ax.plot(x3[0,:], x3[1,:], x3[2,:], label=r'$Hyperion_b$')
ax.view_init(30, 60)
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
plt.legend()
plt.savefig('P1a.png')


# In[11]:


N = 100
fig = plt.figure(figsize=global_fig_size, dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Orbital Motion')
plot0, = ax.plot([],[],[], label=r'$Saturn$')
plot1, = ax.plot([],[],[], label=r'$Titan$')
plot2, = ax.plot([],[],[], label=r'$Hyperion_a$')
plot3, = ax.plot([],[],[], label=r'$Hyperion_b$')
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_zlim(min_z, max_z)
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
plt.legend()
def animate(i):
    plot0.set_data_3d(x0[0,:], x0[1,:], x0[2,:])
    plot1.set_data_3d(x1[0,:], x1[1,:], x1[2,:])
    plot2.set_data_3d(x2[0,:], x2[1,:], x2[2,:])
    plot3.set_data_3d(x3[0,:], x3[1,:], x3[2,:])
    ax.view_init(30, 360*i/N)
    return plot0, plot1, plot2, plot3   
ani = FuncAnimation(fig, animate, N, interval=100, blit=True)
ani.save('P1a.avi')


# In[12]:


fig = plt.figure(figsize=global_fig_size, dpi=200)
ax = fig.add_subplot(111, projection='3d')
scatter0 = ax.scatter([],[],[], label=r'$Hyperion_a$', color='red')
scatter1 = ax.scatter([],[],[], label=r'$Hyperion_b$', color='blue')
plot, = ax.plot([],[],[], color='purple')
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_zlim(min_z, max_z)
ax.view_init(30, 60)
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_title('Dumbbell Axis Motion')
plt.legend()
def animate(i):
    scatter0._offsets3d = np.hsplit(x2[:,i],3)
    scatter1._offsets3d = np.hsplit(x3[:,i],3)
    plot.set_data_3d((x2[0,i],x3[0,i]),(x2[1,i],x3[1,i]),(x2[2,i],x3[2,i]))
    return scatter0, scatter1, plot
ani = FuncAnimation(fig, animate, t.size, interval=50, blit=True)
ani.save('P1b.avi')


# In[13]:


permutations = product([0.5 * dtheta, dtheta], [0.5 * dphi, dphi], [0.25, 0.5, 0.75], [1.0, np.sqrt(2.0)])

for i, p in enumerate(permutations):
    x0, x1, x2, x3, v0, v1, v2, v3, t = run_simulation(theta, p[0], phi, p[1], fps, duration, k, G, alpha=p[2], vel_boost=p[3])
    vec = x3 - x2
    _, _phi, _theta = cart2sph(*vec)
    fig = plt.figure(figsize=global_fig_size, dpi=200);
    ax = fig.add_subplot(111);
    ax.set_title(rf'$\dot\theta_0={p[0]:.4f}$, $\dot\phi_0={p[1]:.4f}$, $\alpha={p[2]:.2f}$, $||v||={p[3]:.4f}\times v_0$');
    ax.plot(t, _theta, label=r'$\theta$');
    ax.plot(t,   _phi, label=r'$\phi$');
    ax.set_xlabel('t [d]');
    ax.set_ylabel('[rad]');
    ax.grid();
    ax.legend();
    plt.savefig(f'P1c/P1c_{str(i).zfill(3)}.png');
    plt.close()


# ## Problem 2
# 
# Now modify the starter code (or create your own program) to study the “butterfly effect”, i.e., follow the evolution from two slightly different initial conditions for the dumbbell axis. Track both the angular orientation difference and angular velocity difference for the same circular and elliptic orbits as in (1), including the real Hyperion orbit. Use the results to further substantiate the conclusions you reached in (1), and calculate the Lyapunov exponents for each case. 

# In[15]:


permutations = product([0.999, 0.99999], [0.5], [1.0, np.sqrt(2.0)])

for i, p in enumerate(permutations):
    _, _, x2a, x3a, _, _, v2a, v3a, ta = run_simulation(p[0]*theta, dtheta,      phi, dphi, fps, duration, k, G, alpha=p[1], vel_boost=p[2])
    _, _, x2b, x3b, _, _, v2b, v3b, tb = run_simulation(     theta, dtheta, p[0]*phi, dphi, fps, duration, k, G, alpha=p[1], vel_boost=p[2])
    _, _, x2c, x3c, _, _, v2c, v3c, tc = run_simulation(     theta, dtheta,      phi, dphi, fps, duration, k, G, alpha=p[1], vel_boost=p[2])
    
    vec_x_a, vec_v_a = x3a - x2a, v3a - v2a
    vec_x_b, vec_v_b = x3b - x2b, v3b - v2b
    vec_x_c, vec_v_c = x3c - x2c, v3c - v2c
    
    (_, phi_a, theta_a), (_, dphi_a, dtheta_a) = cart2sph(*vec_x_a), cart2sph(*vec_v_a)
    (_, phi_b, theta_b), (_, dphi_b, dtheta_b) = cart2sph(*vec_x_b), cart2sph(*vec_v_b)
    (_, phi_c, theta_c), (_, dphi_c, dtheta_c) = cart2sph(*vec_x_c), cart2sph(*vec_v_c)
    
    delta_phi,   delta_dphi   = phi_c   - phi_b,   dphi_c   - dphi_b
    delta_theta, delta_dtheta = theta_c - theta_a, dtheta_c - dtheta_a
    
    delta_phi   /= delta_phi[1];   delta_dphi   /= delta_dphi[1]
    delta_theta /= delta_theta[1]; delta_dtheta /= delta_dtheta[1]
    
    lpnv_phi,   lpnv_dphi   = np.log(delta_phi),   np.log(delta_dphi)
    lpnv_theta, lpnv_dtheta = np.log(delta_theta), np.log(delta_dtheta)
    
    lpnv_theta_peak_idx, lpnv_dtheta_peak_idx = find_peaks(lpnv_theta)[0], find_peaks(lpnv_dtheta)[0]
    lpnv_phi_peak_idx,   lpnv_dphi_peak_idx   = find_peaks(lpnv_phi)[0],   find_peaks(lpnv_dphi)[0]
    
    lpnv_theta_peak_t, lpnv_dtheta_peak_t = np.take_along_axis(ta, lpnv_theta_peak_idx, 0), np.take_along_axis(ta, lpnv_dtheta_peak_idx, 0)
    lpnv_phi_peak_t,   lpnv_dphi_peak_t   = np.take_along_axis(tb,   lpnv_phi_peak_idx, 0), np.take_along_axis(tb,   lpnv_dphi_peak_idx, 0)
    
    lpnv_theta_peaks, lpnv_dtheta_peaks = np.take_along_axis(lpnv_theta, lpnv_theta_peak_idx, 0), np.take_along_axis(lpnv_dtheta, lpnv_dtheta_peak_idx, 0)
    lpnv_phi_peaks,   lpnv_dphi_peaks   = np.take_along_axis(lpnv_phi,   lpnv_phi_peak_idx,   0), np.take_along_axis(  lpnv_dphi,   lpnv_dphi_peak_idx, 0)
    
    lpnv_theta_fit, lpnv_dtheta_fit = linregress(lpnv_theta_peak_t, lpnv_theta_peaks), linregress(lpnv_dtheta_peak_t, lpnv_dtheta_peaks)
    lpnv_phi_fit,   lpnv_dphi_fit   = linregress(lpnv_phi_peak_t,   lpnv_phi_peaks),   linregress(lpnv_dphi_peak_t,   lpnv_dphi_peaks)


    fig, axs = plt.subplots(2,2, figsize=(2*global_fig_size[0],2*global_fig_size[1]), dpi=200, constrained_layout=True);
    fig.suptitle('Lyapunov Constant');
    
    axs[0,0].set_title(
        rf'$\frac{{\left(\Delta\theta\right)_0}}{{\theta_0}}={p[0]:.4f}$, ' + 
        rf'$\frac{{M_{{H_a}}}}{{M_{{H_b}}}}={p[1]/(1-p[1]):.4f}$, ' + 
        rf'$||v||={p[2]:.4f}\times v_0$');
    axs[0,1].set_title(
        rf'$\frac{{\left(\Delta\dot\theta\right)_0}}{{\dot\theta_0}}={p[0]:.4f}$, ' + 
        rf'$\frac{{M_{{H_a}}}}{{M_{{H_b}}}}={p[1]/(1-p[1]):.4f}$, ' + 
        rf'$||v||={p[2]:.4f}\times v_0$');
    axs[1,0].set_title(
        rf'$\frac{{\left(\Delta\phi\right)_0}}{{\phi_0}}={p[0]:.4f}$, ' + 
        rf'$\frac{{M_{{H_a}}}}{{M_{{H_b}}}}={p[1]/(1-p[1]):.4f}$, ' + 
        rf'$||v||={p[2]:.4f}\times v_0$');
    axs[1,1].set_title(
        rf'$\frac{{\left(\Delta\dot\phi\right)_0}}{{\dot\phi_0}}={p[0]:.4f}$, ' + 
        rf'$\frac{{M_{{H_a}}}}{{M_{{H_b}}}}={p[1]/(1-p[1]):.4f}$, ' + 
        rf'$||v||={p[2]:.4f}\times v_0$');

    axs[0,0].scatter(lpnv_theta_peak_t,  lpnv_theta_peaks, color='red');
    axs[0,1].scatter(lpnv_dtheta_peak_t, lpnv_dtheta_peaks, color='blue');
    axs[1,0].scatter(lpnv_phi_peak_t,    lpnv_phi_peaks, color='green');
    axs[1,1].scatter(lpnv_dphi_peak_t,   lpnv_dphi_peaks, color='purple');

    axs[0,0].plot(ta,  lpnv_theta, color='red', label=r'$\ln(\Delta\theta)$');
    axs[0,1].plot(ta, lpnv_dtheta, color='blue', label=r'$\ln(\Delta\dot\theta)$');
    axs[1,0].plot(tb,    lpnv_phi, color='green', label=r'$\ln(\Delta\phi)$');
    axs[1,1].plot(tb,   lpnv_dphi, color='purple', label=r'$\ln(\Delta\dot\phi)$');

    axs[0,0].plot(ta, lpnv_theta_fit.intercept + lpnv_theta_fit.slope * ta, 
    label=rf'$y={lpnv_theta_fit.slope:.4f}t+{lpnv_theta_fit.intercept:.4f}$'+'\n'+
    rf'$r={lpnv_theta_fit.rvalue:.4f}$', ls='--', color='red');
    axs[0,1].plot(ta, lpnv_dtheta_fit.intercept + lpnv_dtheta_fit.slope * ta, 
    label=rf'$y={lpnv_dtheta_fit.slope:.4f}t+{lpnv_dtheta_fit.intercept:.4f}$'+'\n'+
    rf'$r={lpnv_dtheta_fit.rvalue:.4f}$', ls='--', color='blue');
    axs[1,0].plot(tb, lpnv_phi_fit.intercept + lpnv_phi_fit.slope * ta, 
    label=rf'$y={lpnv_phi_fit.slope:.4f}t+{lpnv_phi_fit.intercept:.4f}$'+'\n'+
    rf'$r={lpnv_phi_fit.rvalue:.4f}$', ls='--', color='green');
    axs[1,1].plot(tb, lpnv_dphi_fit.intercept + lpnv_dphi_fit.slope * ta, 
    label=rf'$y={lpnv_dphi_fit.slope:.4f}t+{lpnv_dphi_fit.intercept:.4f}$'+'\n'+
    rf'$r={lpnv_dphi_fit.rvalue:.4f}$', ls='--', color='purple');
    
    for _ in axs:
        for ax in _:
            ax.set_xlabel('t [d]');
            ax.set_ylabel(r'$\Delta$ [rad]');
            ax.grid();
            ax.legend();

    plt.savefig(f'P2a/P2a_{str(i).zfill(3)}.png');
    plt.close()


# In[ ]:




