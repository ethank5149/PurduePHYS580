%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Vnew, deltaV] = capacitor_update_Jacobi(V);
% This function takes a matrix V and applies Eq 5.10 to it. Only the values
% inside the boundaries are changed. It returns the processed matrix to 
% the calling function, together with the value of delta_V, the total
% accumulated amount by which the elements of the matrix have changed
%
rowSize = size(V,1);
colSize = size(V,2);
%
% preallocate memory (speedup
Vnew = zeros(rowSize,colSize);
deltaV = 0.;
%
% create updated grid, element by element, ignoring boundaries
% this means that at the edges of the grid, Vnew = 0
for j = 2:(colSize - 1);
    for i = 2:(rowSize - 1);
        % if we are NOT on the plates
        if V(i, j)~=1 & V(i, j) ~=-1
            % write updated potential in new grid
            Vnew(i, j) = (V(i-1, j) + V(i+1, j) + V(i, j-1) + V(i,j+1)) * 0.25;
            % update error estimate
            deltaV = deltaV + abs(Vnew(i,j) - V(i,j));
        % otherwise, leave value unchanged
        else
            Vnew(i,j) = V(i,j);
        end
    end
end
%
% normalize to return per-lattice-size error
%
deltaV = deltaV / (rowSize * colSize);
%
end
%%%%%%%%%%%%%%%%%%%%
