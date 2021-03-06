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
K = num_labels;
regular = 0;
% You need to return the following variables correctly 
J = 0;
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta1_size = size(Theta1);
Theta2_size = size(Theta2);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m,1),X];

% For each value in y, it copies that row of the eye matrix into y_matrix. 
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1),a2];
z3 = a2 * Theta2';
output = sigmoid(z3);
for i = 1: K
    J = J + ( y_matrix(:,i)' * log(output(:,i)) + (1-y_matrix(:,i)') * log(1-output(:,i)) );
end
J = (-1*J)/m;

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
for i = 2:(Theta1_size(1,2))
    for j = 1:Theta1_size(1,1)
        regular = regular + Theta1(j,i) * Theta1(j,i);
    end
end
for i = 2:(Theta2_size(1,2))
    for j = 1:Theta2_size(1,1)
        regular = regular + Theta2(j,i) * Theta2(j,i);
    end
end

regular = regular * lambda / (2*m);

J = J + regular;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%m = the number of training examples

%n = the number of training features, including the initial bias unit.

%h = the number of units in the hidden layer - NOT including the bias unit

%K = the number of output classifications

 % 1.��3 or d3 is the difference between a3 and the y_matrix. 
 % The dimensions are the same as both, (m x K).
delta_3 = output - y_matrix  
% 2. z2 came from the forward propagation process 
% it's the product of a1 and Theta1, prior to applying the sigmoid() function. 
% Dimensions are (m x n) ? (n x h) --> (m x h)
z2 = X * Theta1';  % m *h 

% 3.��2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. 
% d2 is the product of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. 
% The size is (m x r) ? (r x h) --> (m x h). The size is the same as z2, as must be.
Theta_2 = Theta2(:,2:end);  % (K x h)
delta_2 = delta_3 * Theta_2 .* sigmoidGradient(z2);

% 4. ��1 or Delta1 is the product of d2 and a1. The size is (h x m) ? (m x n) --> (h x n)

Delta1 = Delta1 + delta_2' * X;

% 5.��2 or Delta2 is the product of d3 and a2. The size is (r x m) ? (m x [h+1]) --> (r x [h+1])

Delta2 = Delta2  + delta_3' * a2;

% 7: Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

Theta1_regular = Theta1;
Theta2_regular = Theta2;


% Regularized Neural Networks
% Note that you should not be regularizing the first column of ��(l) which
% is used for the bias term
Theta1_regular(:,1) = 0;
Theta2_regular(:,1) = 0;

Theta1_grad = Theta1_grad + lambda/m * Theta1_regular;
Theta2_grad = Theta2_grad + lambda/m * Theta2_regular;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
