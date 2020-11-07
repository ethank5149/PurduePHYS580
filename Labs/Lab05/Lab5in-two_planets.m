%
% 2 planet simulation of Jupiter, Earth around the Sun. Use Euler Cromer method
% based on 'Computational Physics' book by N Giordano and H Nakanishi
% Section 4.4
% by Kevin Berwick, modified by H. Nakanishi and D. Molnar
% This program treats the Sun as immobile at the origin.
%
clear all;
%
% Read physics parameters
%
% The star can be the Sun
%   Mass of the Sun is  2x10^{30} kg
%
% Planet 1 (can be Earth)
%   Mass of Earth is 6x10^{24} kg
%   Initial position of Earth is (1,0) in AU
%   Initial velocity of Earth is (0,2*pi) in AU/Yr
%
disp       ('Planet 1: ');
mE  = input('  m_1 / M_star: ');
xE  = input('  x_0 [AU]: ');
yE  = input('  y_0 [AU]: ');
vxE = input('  vx_0 [AU/yr]: ');
vyE = input('  vy_0 [AU/yr]: ');
%
% Planet 2 (can be Jupiter)
%   Mass of Jupiter is 1.9x10^{27} kg
%   Initial position of Jupiter is (5.2,0) in AU
%   Initial velocity of Jupiter is (0,2.7549) in AU/Yr
%   This is 2*pi*5.2 AU/11.85 years = 2.75 AU/year
%
disp       ('Planet 2: ');
mJ  = input('  m_2 / M_star: ');
xJ  = input('  x_0: ');
yJ  = input('  y_0: ');
vxJ = input('  vx_0: ');
vyJ = input('  vy_0: ');
%
% Read and set computational parameters
%
t_end = input('simulation end time [yr]: ');
dt    = input('dt [yr]: ');
nsteps = floor(t_end / dt + 0.5);
t = 0;
%
% set up trajectory plot
clf;
plot(xE, yE, 'dg', xJ, yJ, '+b', 0, 0, 'or');
axis([-7 7 -7 7]);
xlabel('x [AU]');  ylabel('y [AU]');
hold on;
path1 = animatedline(xE, yE, 'Color', 'g');
path2 = animatedline(xJ, yJ, 'Color', 'b');
%
% evolve equations of motion via Euler-Cromer
%
GM = 4 * pi^2;
for i = 1:nsteps
    % Earth-Sun, Jupiter-Sun, and Earth-Jupiter distances
	  rES = sqrt(xE^2 + yE^2);
	  rJS = sqrt(xJ^2 + yJ^2);
	  rEJ = sqrt((xE-xJ)^2 + (yE-yJ)^2);
    % update Earth's velocity
	  vxE = vxE - GM * xE*dt / rES^3 - GM*mJ * (xE-xJ)*dt / rEJ^3;
	  vyE = vyE - GM * yE*dt / rES^3 - GM*mJ * (yE-yJ)*dt / rEJ^3;
    % update Jupiter's velocity
	  vxJ = vxJ - GM * xJ*dt / rJS^3 - GM*mE * (xJ-xE)*dt / rEJ^3;
	  vyJ = vyJ - GM * yJ*dt / rJS^3 - GM*mE * (yJ-yE)*dt / rEJ^3;
    % update positions and time
	  xE = xE + vxE * dt;
	  yE = yE + vyE * dt;
  	xJ = xJ + vxJ * dt;
  	yJ = yJ + vyJ * dt;
  	t  = t + dt;
    % add new positions to trajectories
  	addpoints(path1, xE, yE);
	  addpoints(path2, xJ, yJ);
    % update plot with new data
	  title(['Simulation - Star (fixed) and 2 planets, ',num2str(t),' years']);
    drawnow;
end;
%
%
%