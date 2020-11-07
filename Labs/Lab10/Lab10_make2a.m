
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%     subroutine "make" to make a percolation cluster, label tree version
%
%     code from Hisao Nakanishi, modified by Denes Molnar
%
%  list:  sites of largest cluster
%  list2: sites of all other clusters
%  numb:  number of occupied sites
%  nmax:  size of largest cluster
%  nc:    total number of clusters
%  nna:   sorted array of cluster sizes
%  nnb:   number of clusters (nnb(i)) of given size (nna(i))
%  vc:    number of distinct cluster sizes  (=numel(nna))
%
% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
function [list,list2,numb,nmax,nc,nna,nnb,vc] = Lab10_make2a(p,Nedge);

latt  = zeros(Nedge, Nedge);
list  = zeros(2, Nedge^2);     % largest cluster points
list2 = zeros(2, Nedge^2);     % rest of clusters points

%==============================================================================
%   Start generating clusters in Nedge x Nedge grid.
%==============================================================================
%   Generate the first column (on lattice space) from top to bottom.
%------------------------------------------------------------------------------
%   In this program, column refers to (i,.) for fixed i even though this is
%   actually the opposite of the order of matrix indices.  This is simply
%   because it is easier to interpret indices as (x,y).
%------------------------------------------------------------------------------
% 
% Start with the first 'column' to generate. here, column refers to the
% first line of sites in y-direction in our thinking of the indices of 
% the matrix latt(i,j) as latt(x,y). In Matlab, actually fixing the first
% index and running through the second index means a column in the matrix.
% So, perhaps we should have called this the first 'row' instead. But then,
% when we plot the maximal cluster with 'scatter', we do use x, and y for 
% horizontal and vertical direction. So there, we do mean a 'column'.
%
ic = 0;
if rand <= p       % if 'top left' site (1,1) is to be occupied
   ic = ic + 1;    %   increment cluster number this site belongs to
   latt(1,1) = ic; %   save that number (ic=1) in the array latt
   label(ic) = 1;  %   start the label tree for the culster numbers
                   %   this cluster (ic=1) has 1 site in it (so far)
else               % else if (1,1) stays empty
   latt(1,1) = 0;  %   empty sites have zero in that location in latt
end
for j=2:Nedge
   if rand <= p         % if site (1,j) is to be occupied
      if latt(1,j-1) > 0                      % site above occ & already counted
         latt(1,j) = latt(1,j-1);             % use the same label here
         label(latt(1,j))=label(latt(1,j))+1; % label tree for this cluster is
                                              % incremented to add this site
      else               % if this site is occupied but site above was empty
         ic = ic + 1;    %   new cluster number needed
         latt(1,j) = ic; %   save the new cluster number in the array latt (for speed)
         label(ic) = 1;  %   this cluster has 1 site in it (so far)
      end
   else                % else if (1,j) stays empty
         latt(1,j)=0;
   end
end %j

%   Generate column 2 through column "Nedge"; each one from top to bottom.
%
for i=2:Nedge
   % The top site of each column.
   %  
   if rand <= p         % if (i,1) to be occupied, only check the site to the left for conn.
      if latt(i-1,1) > 0     % if site to left (L) is occupied.
         ip = latt(i-1,1);   %   the current number for the cluster that L belongs to
         while label(ip) < 0 %   check label tree to see if this label is a pointer 
            ip = -label(ip); %   get the label to which this label points to
         end
         latt(i,1) = ip;            % save ip = proper label (cluster number) into latt
         label(ip) = label(ip) + 1; % label tree for the proper label is the clust size
      else                   % if site to left (L) is empty
         ic = ic + 1;               % need a new label
         latt(i,1) = ic;
         label(ic) = 1;
      end
   else                 % if (i,1) stays empty
      latt(i,1) = 0;
   end
   % The rest of the column.
   %
   for j=2:Nedge;
      if rand <= p    % if (i,j) to be occupied
         if latt(i,j-1) * latt(i-1,j) > 0 % sites above and to left are both occupied
                                          % the two existing clusters merge here at 
                                          % the newly occupied site
            ip = latt(i-1,j);
            iq = latt(i,j-1);
            while label(ip) < 0      % find the proper label of site to left (L)
               ip = -label(ip);
            end
            while label(iq) < 0      % find the proper label of site above (U)
               iq = -label(iq);
            end
            if ip == iq              % if the proper labels are already the same
               latt(i,j) = ip;           % use this label for the new site
               label(ip) = label(ip) + 1;  % increment the cluster size
            else                     % if proper labels are different,
               jmin = min(ip,iq);          % find the lower value and make it the
               jmax = max(ip,iq);          % new proper label for the merged cluster
               latt(i,j) = jmin;         
               label(jmin) = label(jmin) + label(jmax) + 1; % update the its size
               label(jmax) = -jmin;      % label tree makes the other label point to it
            end
         elseif (latt(i,j-1)+latt(i-1,j)>0)  % if only one of U or L is occupied
            imx = max(latt(i,j-1),latt(i-1,j));   % imx is the label of the occ. site
            ip = imx;                             % so the current site adds to this clust
            while label(ip) < 0                 % find the proper label for this cluster
               ip = -label(ip);
            end
            latt(i,j) = ip;                     % save that in latt
            label(ip) = label(ip) + 1;          % increment the size of this cluster
         else                               % if L and U are both empty
            ic = ic + 1;                        % new cluster started
            latt(i,j) = ic;
            label(ic) = 1;
         end
      else     % elseif (i,j) stays empty
         latt(i,j) = 0;
      end
   end %j
 
end %i

%==============================================================================
%   Finished generation.  Find the cluster number of the largest cluster.
%==============================================================================

nmax = 0;
numb = 0;
id = 0;
for i = 1:ic
   if label(i) > 0
      numb = numb + label(i);
      if label(i) > nmax       % there could be multiple maximal clusters
         nmax = label(i);      
         id = i;               % only recording one of the max cluster's index
      end
   end
end
     
%==============================================================================
%   Set up list array for the maximal cluster (one of them if multiple clusters
%   of the same maximum size exist)
%==============================================================================
%
%   First put each site of the maximal cluster into "list"
%
n = 0;
m = 0;
for i=1:Nedge
   for j=1:Nedge
      ip = latt(i,j);
      if ip <= 0          % this site is empty
         continue;        % go to next index j (to next site)
      end
      while label(ip) < 0  % find the proper label of this occupied site
         ip = -label(ip);
      end
      if ip == id         % the proper label matches that of the maximal cluster
         n = n + 1;       % increment size index 'n'
         list(1,n) = i;   % save the x- and y-coorinates of this site in 'list'
         list(2,n) = j;
      else
         m = m + 1;       % this site is occupied but not part of the max cluster
         list2(1,m) = i;  % save the coordinates into another matrix 'list2'
         list2(2,m) = j;
      end
    end %j
end %i

%==============================================================================
%   Finished setting up the output array.
%==============================================================================
%==============================================================================
%   Set up "nna" (size) and "nnb" (number of clusters of that size)
%==============================================================================

j = 0;
nc = 0;
vc = 0;
for i = 1:ic         % go through all cluster labels (proper/improper)
   s = label(i);          % array 'label' has all needed info
   if s > 0                  % 'i' is a proper label for a cluster of size 's'
      nc = nc + 1;               % increment total number of clusters 'nc'
      found = 0;
      for k = 1:j            % check if this size (=s) cluster has been found yet
         if nna(k) == s        % it has been found before with index 'k'
            nnb(k) = nnb(k) + 1; % increment the cluster count for this 'k'
            found = 1;         % set flag for 'nna'/'nnb' should not be re-initialized
            break;             % go to the next cluster number 'i'
         end
      end %k
      if found == 0      % first time this size 's' cluster has been found
         vc = vc + 1;       % increment the counter of distinct cluster sizes
         j = j + 1;         % increment cluster size index 'j'
         nna(j) = s;        % record the size of this 'j'
         nnb(j) = 1;        % initialize the cluster count for this 'j' to be 1
      end
   end %s>0
end %i

[nna,I] = sort(nna); % sort array 'nna' in ascending values and put original
                     % corresponding indices into array 'I'
nnb = nnb(I);        % re-index array 'nnb' using 'I' so that new nna and nnb
                     % still have the proper correspondence
%==============================================================================
%   Finished setting up the output arrays.
%==============================================================================

