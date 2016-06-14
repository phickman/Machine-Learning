function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


% If you don't want matrix multiplication, you need to add the \dot" notation
% to specify this to Octave/MATLAB. For example, A*B does a matrix
% multiply, while A.*B does an element-wise multiplication.

hyp = X * theta;
sq_err = (hyp - y) .^ 2;
J = sum(sq_err) / (2 * m);


% =========================================================================

end
