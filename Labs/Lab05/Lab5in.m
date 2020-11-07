%
% Planetary orbit using second order Euler-Cromer method.
% by Kevin Berwick, slightly edited by H. Nakanishi and D. Molnar
% based on 'Computational Physics' book by N Giordano and H Nakanishi
% Section 4.1
%
% AU-year units (for Solar System)
%
%    solves d^2r/dt^2 = - G*M r / r^3 from given inittial conditions
%
clear all;
%
% set initial conditions
x  = input('x_0 [AU]: ');
y  = input('y_0 [AU]: ');
vx = input('vx_0 [AU/yr]: ');
vy = input('vy_0 [AU/yr]: ');
t  = 0.;
% set computational parameters
t_end  = input('t_end [yr]: ');
dt     = input('dt [yr]: ');
nsteps = floor(t_end / dt + 0.5);
%
% plot the Sun at the origin and keep it for later plots
%
clf;
plot(0, 0, 'oy', 'MarkerSize', 50, 'MarkerFaceColor', 'yellow');
h = animatedline(x, y, 'Color', 'b');
axis([-1.5 1.5 -1.5 1.5]);
xlabel('x [AU]');
ylabel('y [AU]');
%hold on;
%
% Euler-Cromer time evolution
GM = 4 * pi^2;
for i = 1:nsteps
    % plot point
    addpoints(h, x, y);
    title([num2str(t),' years']);
    drawnow;
    %
    % update
    r = sqrt(x^2+y^2);
    % compute new position (x,y), velocity (vx,vy)
    vx_new = vx - GM * x * dt / r^3;
    vy_new = vy - GM * y * dt / r^3;
    x_new  = x  + vx_new * dt;
    y_new  = y  + vy_new * dt;
    % update positionss and velocities with new values, and time too
    vx = vx_new; 
    vy = vy_new;
    x = x_new;
    y = y_new;
    t = t + dt;
end
%
%%
