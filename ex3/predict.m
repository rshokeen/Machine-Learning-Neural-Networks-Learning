function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% Theta1 has size 25 by 401. They are already trained. They are sized for neural network with 25 units in second layer.
% Theta2 has size 10 by 26. They are already trained. They corresspond to 10 output units (or 10 digit classes) in third layer.

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


X = [ones(m, 1) X]; % Add ones to the X data matrix

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

a2 = sigmoid(X * Theta1'); %gives 5000 by 25 matrix


a2 = [ones(m, 1) a2]; % Add ones to the a2 data matrix and provides 5000 by 26 matrix


a3 = sigmoid(a2 * Theta2'); %5000 by 26 multiplied with 26 by 10 matrix provides 5000 by 10 matrix

[value, index] = max(a3, [], 2); %find the max value and index for each of 5000 example 

p = index; % assign highest value's index (anywhere between 1 - 10)





% =========================================================================


end
