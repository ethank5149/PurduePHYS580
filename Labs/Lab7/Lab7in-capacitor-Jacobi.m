%
% Jacobi method for solving the Laplace equation in 2D
% based on 'Computational Physics' book by N Giordano and H Nakanishi
% Section 5.1
% code by Kevin Berwick, revised by Hisao Nakanishi and Denes Molnar
%
% Capacitor geometry, in 2D cut perpendicular to plates
% Plates are parallel to y axis and of width a*L, separated by distance b*L 
%   (L is arbitrary)
% Computational box in x,y: [-L/2,L/2]x[-L/2,L/2]
%
% Boundary conditions: V =+V_0 on left plate, -V_0 on right plate
%                      V=0 at the box boundary
%
clear all;
%
% Read parameters
%
% physical parameters
a = input('width of plates as fraction of box size (a): ');
b = input('gap between plates as fraction of box size (b): = ');
%
% computational parameters
dx = input('grid spacing as fraction of box size (dx): ');
acc = input('average error as fraction of V_0 (acc): ');
%
% Convert (x,y) to positive integer row/column indices
% The correspondence with geometry is (x,y) geometry <=> [-y x] matrix
% 
% origin of the grid is at (M+1, M+1)
%
% NOTE: rounding is involved
%       if your a,b do not correspond to plates falling right onto grid points
%       then the plates will, effectively, be relocated to be on grid points
%
M = floor(0.5 / dx + 0.5);
dx = 1. / (2 * M);
N = 2 * M + 1;   # linear grid size
aM = floor(a * M + 0.5);
bM = floor(b * M + 0.5);
V = zeros(N);
for i = (M + 1 - aM):(M + 1 + aM)
    V(i, M + 1 + bM) = -1;
    V(i, M + 1 - bM) = 1;
end
%
% Run update routine once and obtain the initial local error delta_V
% Update routine calculates new V, which we move back to V
%
[Vnew, deltaV] = Lab7_capacitor_update_Jacobi(V);
V = Vnew;
%
% Initialise loop counter and the mesh for potential surface
% mesh ojbect handle is obtained for the animation purpose
%
clf;
iter = 1;
[X, Y] = meshgrid(-0.5:dx:0.5);
m = mesh(X, Y, V, 'Facecolor', 'interp');
s = sprintf('Potential Surface, acc = %0.5e5, iter = %d', acc, iter);
axis([-0.5 0.5 -0.5 0.5 -1 1]);
xlabel('X in units of L');
ylabel('Y in units of L');
zlabel('V in units of V_0');
hold on;
drawnow;
%
% Iterate, until error goal is reached
%
while (deltaV > acc)
    [Vnew, deltaV]=capacitor_update_Jacobi(V);
    iter = iter + 1;
    V = Vnew;
    % redraw the surface by replacing the original surface (change its property)
    set(m, 'ZData', V);
    s = sprintf('Potential Surface, acc = %0.5e, iter = %d', deltaV, iter);
    title(s);
    drawnow;
end
%
%
