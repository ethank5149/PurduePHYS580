%
% Cannon ball trajectory computed via Euler approximation
% 
% assumes: i) drag force F = - B_2 * v^2
%          ii) isothermal air density model: rho(y) = rho(0) * exp(-y / Y0)
%
clear all;
%
% Initialize physical parameters (units are in [] brackets)
%   v0 [m/s], theta [degrees], B2/m  [1/m],  Y0 [m]
% some reasonable choices are B2/m = 4x10^{-5}, y0 = 10^4.
%
v0 = input('initial speed [m/s]: ');
theta = input('shooting angle [degrees]: ');
B2_m = input('B2/m [1/m]: ');
Y0 = input('characteristic length in atmospheric pressure [m]: ');
g = 9.8;   % gravitational acceleration [m/s^2] - FIXED
%
% Initial conditions
%
x(1) = 0;
y(1) = 0;
vx = v0 * cos(theta/180*pi);
vy = v0 * sin(theta/180*pi);
% max range of shell, in vaccuum (for automatic horizontal plot range)
maxr = v0*v0/g;
% flight time to maxr, in vacuum (for automatic calculation of end time)
maxt = maxr/vx;
%
% Initialize computational parameters; dt [s], number of points
%
dt = input('dt [s]: ');
nsteps = round(maxt/dt);
%
% Compute evolution using Euler method
%
for i = 1:nsteps
   x(i+1) = x(i) + vx*dt;
   y(i+1) = y(i) + vy*dt;
   f = B2_m * (vx^2+vy^2)^0.5 * exp(-y(i)/Y0);  % isothermal model of air
   vx = vx - f * vx * dt;
   vy = vy - g * dt - f * vy * dt;
   if (y(i+1) <= 0)  % stop calculation if at i+1 it is below ground
     break
 end
end
%
% Interpolate between steps i and i+1 linearly to find the range
%
xmax = (y(i+1) * x(i) - y(i) * x(i+1) ) / (y(i+1) - y(i));
range = round(xmax);   % rounded value for plot title
x(i+1) = xmax;
y(i+1) = 0;
%
% Plot y vs x
%
plot(x,y,'r','LineWidth',3);
axis([0 maxr*1.05 0 inf]);  % x & y ranges
xlabel('x [m]')
ylabel('y [m]')
title({['dt = ',num2str(dt),', v0 = ',num2str(v0),', B2/m = ',...
    num2str(B2_m),', h = ',num2str(Y0),', theta = ',num2str(theta),...
	', Range = ',num2str(range)];'Euler approximation'});
% uncomment line below if you want to draw over previous plot(s)
%hold on;
