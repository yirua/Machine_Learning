function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
%fprintf('X is: \n');
%X
%fprintf('centroids is: \n');
%centroids
m = size(X,1);
distance = zeros(m, K);
fun = @(a,b) (a-b).^2;
for i=1:K
 sample_vec= zeros(m,1);
 for j=1:m
	
	sample_vec(j) = sum(bsxfun(fun, X(j,:),centroids(i,:)));
 end
%sample_vec() = sum(bsxfun(fun, X,centroids(i,:)));
 %fprintf('sample_vec is: \n');
 %sample_vec;
 distance(:, i) = sample_vec;% column i, all rows, all X samples with centroids(i) distance....
 
end
%fprintf('disance matrix: \n');
distance;

%fprintf('The min value and index: \n');
[value, idx]=min(distance');
idx=idx(:);




% =============================================================

end

