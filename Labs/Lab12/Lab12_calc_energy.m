% calculate current energy (potential + kinetic)
% also compute temperature via equipartition
%
% code from Hisao Nakanishi, modified by Denes Molnar

function [E,Epot,T] = Lab12_calc_energy(x, y, vx, vy, Npart, L);

Ekin = 0;   % kinetic energy
Epot = 0;   % potential energy
% loop through particles
for i = 1:Npart
   % update kinetic energy
   Ekin = Ekin + 0.5 * (vx(i)^2 +vy(i)^2);
   % take all unique pairs (j > i) for potential energy
   for j = (i+1):Npart
      dx = x(1,j) - x(1,i);
      dy = y(1,j) - y(1,i);
      if abs(dx) > 0.5 * L     % use nearest separation rule
         dx = dx - sign(dx) * L;
      end
      if abs(dy) > 0.5 * L
         dy = dy - sign(dy) * L;
      end
      r = sqrt(dx^2 + dy^2);
      if r < 3                % cut interaction off at r > 3 sigma
         invr6 = 1. / r^6;         % shorthand for 1/r^6
         invr12 = invr6 * invr6;   % shorthand for 1/r^12
         Epot = Epot + 4 * (invr12 - invr6);
      end
   end %j
end %i
E = Ekin + Epot;        % total energy
T = Ekin / Npart;       % equipartition in 2D
%
%END
