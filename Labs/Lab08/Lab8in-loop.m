% magnetic field of a current loop in the x-y plane, centered at the origin
%
% by Denes Molnar, based on codes from Hisao Nakanishi and
%
%
% Initialize variables
%
clear;
%
disp('Field of loop in x-y plane, evaluated over x-z-plane rectangle of size 2L');
R = input('loop radius (R) in units of L: ');

% Set computational parameters
%
% Nphi is the number of intervals for the phi integration on [0,2pi]
%
% output is produced on a grid centered at the origin, which has
% N cells in each direction (counted from the ORIGIN)
% i.e., there are in total (2N+1) grid points along x, y, z
%
% minimal setting might be N=5, Nphi=20
%
Nphi = input('phi steps along loop (Nphi): ');
N = input('grid cells from ORIGIN to edge of rectangle (N): ');
%
%
% Calculate field for grid points in the x-z plane [(2N+1) x (2N+1) matrix] 
%
bxVals = zeros(2 * N + 1);
byVals = zeros(2 * N + 1);
bzVals = zeros(2 * N + 1);
%
dx = 2. / (2 * N);
for i = 1:(2 * N + 1)             % x direction
   for k = 1:(2 * N + 1)          % z direction
      x = dx * (i - N - 1);
      z = dx * (k - N - 1); 
      % calculate B at each observation point in x-z plane
      [bx, by, bz] = Lab8_loop_calculate_B(x, 0., z, R, Nphi);
      % store result - NOTE: (k,i) order used, i.e., (z,x)
      bxVals(k,i) = bx;
      byVals(k,i) = by;
      bzVals(k,i) = bz;
   end
end
%
% plot the B vector field in xz-plane
clf;
x = [-1:dx:1];
z = [-1:dx:1];
[X, Z] = meshgrid(x, z);
quiver(X, Z, bxVals, bzVals, 'r', 'LineWidth', 3);
axis([-1 1 -1 1]);
title(['Angular intervals = ', num2str(Nphi)]);
xlabel('X (in units of L)');
ylabel('Z (in units of L)');
pause;
hold on;
%
% show the location of loops
cx = [R -R];
cz = [0 0];
plot(cx, cz, 'b.', 'MarkerSize', 50);
pause;
%
% plot the B along the z-axis. first index of field3 is the z-values
%
% B_z and B_x vs z
clf;
plot(z, bzVals(:,N+1), 'rx', 'LineWidth', 3, ...
     z, bxVals(:,N+1), 'b+', 'LineWidth', 3);
xlabel('z');
ylabel('B_z or B_x');
title('Magnetic field from a current loop');
legend('B_z vs z at x=y=0', 'B_x vs z at x=y=0');
hold on;
%
% superimpose theoretical result for the coil
% (NOTE: beware, this is not a meaningful test of accuracy...)
%
z2 = [-1:dx*0.1:1];
b2 = zeros(numel(z2), 1);
for i = 1:numel(z2) 
    b2(i) = 2 * pi * R^2 / (z2(i)^2 + R^2)^1.5;
end
plot(z2, b2, 'r', z2, 0*b2, 'b');
%

%
%
%END
