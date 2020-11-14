% code by Hisao Nakanishi, modified by Denes Molnar
%
% do the real work here at temperature T and field H
% spin color updates on scatter plots for graphic display
%

function [spin,E,m] = Lab11_calculate_spin(spin,N,T,H,E,m,sz);

% one sweep over NxN lattice along each row and column
for i = 1:N
   for j = 1:N
      % compute energy change if we flip spin at (i,j)
      % use sum neighboring spins, periodic boundary conditions
      sum = spin(mod(i,N)+ 1,j) + spin(i,mod(j,N)+1) ...
             + spin(N - mod(1-i,N),j) + spin(i,N -mod(1-j,N));
      deltaE = 2 * spin(i,j) * (sum + H);
      % always flip if it lowers the energy
      % otherwise, flip with Metropolis probability
      if (deltaE < 0) || (exp(-deltaE / T) > rand)         
         spin(i,j) = -spin(i,j);
         m = m + 2 * spin(i,j);
         E = E + deltaE;
         flip = 1;
      else
         flip = 0;
      end
      % visualize flipped spins - comment &&false out in production runs
      if flip == 1 %&& false
         if spin(i,j) == -1
            scatter([i], [j], sz, 'b', 'Filled');
         else
            scatter([i], [j], sz, 'r');
         end         
         hold on; 
      end 
   end %j
end %i
%
%END
