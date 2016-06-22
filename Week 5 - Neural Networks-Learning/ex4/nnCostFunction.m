function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% The matrix X contains the examples in rows
% (i.e., X(i,:)' is the i-th training example x(i), expressed as a n x 1
% vector.) When you complete the code in nnCostFunction.m, you will
% need to add the column of 1's to the X matrix. The parameters for each
% unit in the neural network is represented in Theta1 and Theta2 as one
% row. Specifically, the first row of Theta1 corresponds to the first hidden
% unit in the second layer. You can use a for-loop over the examples to
% compute the cost.

% Tip: One handy method for excluding a column of bias units is to use the notation 
% SomeMatrix(:,2:end). This selects all of the rows of a matrix, and omits the entire first column.

% 1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). 
% This is most easily done using an eye() matrix of size num_labels, with 
% vectorized indexing by 'y'. A useful variable name would be "y_matrix", as this...
y_matrix = eye(num_labels)(y,:);

% 2 - Perform the forward propagation:

%% Input Layer
% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m, 1) X];
% z2 equals the product of a1 and Θ1
z2 = Theta1 * a1';

%% Output Layer
% a2 is the result of passing z2 through g()
a2 = sigmoid(z2);
% Then add a column of bias units to a2 (as the first column).
a2 = [ones(1, size(a2, 2)); a2];
% NOTE: Be sure you DON'T add the bias units as a new row of Theta.

%% Output Layer
% z3 equals the product of a2 and Θ2
z3 = Theta2 * a2;
% a3 is the result of passing z3 through g()
a3 = sigmoid(z3);

% Cost Function, non-regularized:

% 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% using a3, your y_matrix, and m (the number of training examples).
% Note that the 'h' argument inside the log() function is exactly a3.
% Cost should be a scalar value. Since y_matrix and a3 are both matrices,
% you need to compute the double-sum.

hyp = a3;

% Remember to use element-wise multiplication with the log() function.
% Also, we're using the natural log, not log10().

% i think ex3 was ok with matrix multiplication because one parameter
% was a vector.  this time both are matrices.
cost_calc = (-y_matrix' .* log(hyp) - (1 - y_matrix') .* log(1 - hyp));
% make cost_calc one column with cost_calc(:) and sum all (need scalar value)
J = (1/m) * sum(cost_calc(:));

% Now you can run ex4.m to check the unregularized cost is correct, 
% then you can submit this portion to the grader.

% Cost Regularization:

% 4 - Compute the regularized component of the cost according to ex4.pdf Page 6, 
% using Θ1 and Θ2 (excluding the Theta columns for the bias units), along with λ, 
% and m. The easiest method to do this is to compute the regularization terms 
% separately, then add them to the unregularized cost from Step 3.

% You can run ex4.m to check the regularized cost, then you can submit this 
% portion to the grader.

% exclude the bias column
temp_theta1 = Theta1(:,2:end);
temp_theta2 = Theta2(:,2:end);

J += (lambda / (2*m)) * (sum(temp_theta1(:).^2) + sum(temp_theta2(:).^2));  % regularised


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
