function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
predict_matrix = X * theta;   % dimension (m*1)
disp(predict_matrix)
error_matrix = (predict_matrix - y ).^2;
error = sum(error_matrix);
theta_regular = theta(2:end,1);

J = 1/(2*m)*error + lambda/(2*m)* (theta_regular' * theta_regular);


grad = (1/m * (predict_matrix - y)'* X)' + lambda/m *theta;

%for j = 0£»
grad(1,1) = 1/m * (predict_matrix - y)'* X(:,1);

% =========================================================================

grad = grad(:);

end
