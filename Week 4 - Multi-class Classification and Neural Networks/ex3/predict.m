function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% Implementation Note: The matrix X contains the examples in rows.
% When you complete the code in predict.m, you will need to add the
% column of 1's to the matrix. The matrices Theta1 and Theta2 contain
% the parameters for each unit in rows. Specifcally, the first row of Theta1
% corresponds to the first hidden unit in the second layer. In Octave/MAT-
% LAB, when you compute z(2) = theta(1)a(1), be sure that you index (and if
% necessary, transpose) X correctly so that you get a(l) as a column vector.



% This is an implementation of the formula in Figure 2 on Page 11 of ex3.pdf.

% 1. Add a column of 1's to X (the first column), and it becomes 'a1'.
a1 = [ones(m, 1) X];

% 2, Multiply by Theta1 and you have 'z2'.
z2 = a1 * Theta1';

% 3. Compute the sigmoid() of 'z2' and add a column of 1's, and it becomes 'a2'
a2 = [ones(m, 1) sigmoid(z2)];

% 4. Multiply by Theta2, compute the sigmoid() and it becomes 'a3'.
a3 = sigmoid(a2 * Theta2');

% 5. Now use the max(a3, [], 2) function to return two vectors - one of the highest 
%    value for each row, and one with its index. Ignore the highest values.
%    Keep the vector of the indexes where the highest values were found. 
%    These are your predictions.
[max_value, p] = max(a3, [], 2);

% Note: When you multiply by the Theta matrices, you'll have to use
% transposition to get a result that is the correct size.

% Note: the predictions must be returned as a column vector - size (m x 1). 
% If you return a row vector, the script will not compute the accuracy correctly.

% ------ dimensions of the variables ---------
% a1 is (m x n), where 'n' is the number of features including the bias unit
% Theta1 is (h x n) where 'h' is the number of hidden units
% a2 is (m x (h + 1))
% Theta2 is (c x (h + 1)), where 'c' is the number of labels.
% a3 is (m x c)
% p is a vector of size (m x 1)



% =========================================================================


end
