
% move forward one time step
% x(i,n), y(i,n) = position of particle i (n = 1: old, 2: current position)
% (x_new, y_new) = new position
% vx,vy = velocity components
% Npart = number of particles
% L     = size of box (use periodic boundary conditions)   
% dt    = time step
%
% code by Hisao Nakanishi, modified by Denes Molnar

function [x,y,vx,vy,r2,rp2,x1,y1,x2,y2] = Lab12_updatemd(x, y, vx, vy, Npart, L, dt, x1, y1, x2, y2)

x_new = zeros(Npart, 1);
y_new = zeros(Npart, 1);
for i = 1:Npart
   fx = 0;
   fy = 0;
   for j = 1:Npart
      if j~=i     % skip i == j
         dx = x(2,i) - x(2,j);
         dy = y(2,i) - y(2,j);
         if abs(dx) > 0.5 * L       % using nearest separation rule
            dx = dx - sign(dx) * L;
         end
         if abs(dy) > 0.5 * L
            dy = dy - sign(dy) * L;
         end
         r = sqrt(dx^2 + dy^2);
         if r < 3                 % cut off interaction at r > 3*sigma
            fijMagn = 24 * (2/r^13 - 1/r^7);  % magnitude of f_ij force
            fx = fx + fijMagn * dx/r;
            fy = fy + fijMagn * dy/r;
         end
      end
   end %j

   % Verlet update, also track velocities
   x_new(i) = 2*x(2,i) - x(1,i) + fx * dt^2;
   y_new(i) = 2*y(2,i) - y(1,i) + fy * dt^2;
   vx(i) = (x_new(i) - x(1,i)) / (2*dt);
   vy(i) = (y_new(i) - y(1,i)) / (2*dt);
   % separately track motion of particles 1 & 2
   if i == 1
      x1 = x1 + x_new(1) - x(2,1);
      y1 = y1 + y_new(1) - y(2,1);
      r2 = x1^2 + y1^2;
   elseif i == 2
      x2 = x2 + x_new(2) - x(2,2);
      y2 = y2 + y_new(2) - y(2,2);
      rp2 = (x2-x1)^2 + (y2-y1)^2;
   end
   % enforce periodic boundary conditions
   if x_new(i) < 0
      x_new(i) = x_new(i) + L;
      x(2,i)   = x(2,i)   + L;
   elseif x_new(i) > L 
      x_new(i) = x_new(i) - L;
      x(2,i)   = x(2,i)   - L;
   end
   if y_new(i) < 0
      y_new(i) = y_new(i) + L;
      y(2,i)   = y(2,i)   + L;
   elseif y_new(i) > L 
      y_new(i) = y_new(i) - L;
      y(2,i)   = y(2,i)   - L;
   end
end %i

for i = 1:Npart         % update current and old values
   x(1,i) = x(2,i);                          
   x(2,i) = x_new(i);
   y(1,i) = y(2,i);
   y(2,i) = y_new(i);
end
%
%END

