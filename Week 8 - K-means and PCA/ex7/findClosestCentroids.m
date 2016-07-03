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


% easy way: iterate every example
%for i = 1:length(X)
%  xi = X(i,:);
%  diff = sum((xi-centroids).^2');
%  [m midx] = min(diff);
  % find closest centroid
%  idx(i) = midx;
%end


% followed tutorial => https://www.coursera.org/learn/machine-learning/discussions/weeks/8/threads/ncYc-ddQEeWaURKFEvfOjQ
% number of training examples by number of centroids
distance = zeros(size(X,1), K);

for i = 1:K
  % calculate distance of all X's to this centroid
  diffs = bsxfun(@minus, X, centroids(i,:));
  % calculate sum of the squared differences
  distance(:,i) = sum(diffs.^2, 2);
end

% identify the column (centroid) with the minimum sum of squared differences
[mx, idx] = min(distance, [], 2);


% =============================================================

end

