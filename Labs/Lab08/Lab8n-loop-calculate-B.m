% calculate the B field at position (x,y,z) for a current loop
% of radius R, lying in the x-y plane centered at the origin
%
% by Denes Molnar, based on code from Hisao Nakanishi
%
function [bx, by, bz] = Lab8_loop_calculate_B(x, y, z, R, Nphi)
   % initialize sums for the integral
   bx = 0.;
   by = 0.;
   bz = 0.;
   % step size for angular integral
   dphi = 2 * pi / Nphi;
   % loop through phi elements
   for i = 1:Nphi
      phi = (i - 1) * dphi;
      # components of line element dl (note, dlz = 0)
      dlx = - R * dphi * sin(phi);    
      dly =   R * dphi * cos(phi); 
      # components of vector from line element to point of observation
      rx = x - R * cos(phi);     
      ry = y - R * sin(phi);
      rz = z; 
      r = sqrt(rx^2 + ry^2 + rz^2);
      % sum contributions dl x r / r^3, but avoid getting close to the wire
      if r > R * 1e-4                       
         bx = bx +  dly * rz / r^3;  
         by = by -  dlx * rz / r^3;
         bz = bz + (dlx * ry - dly * rx) / r^3;
      end
   end  %for-i
end
%
%END
