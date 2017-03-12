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
h= X*theta;%(m*n * n*1)
error = h-y;%(m*1)
error_sqr = error.^2;
q= sum(error_sqr);

J = 1/(2*m)*q;
%J = computeCost(X,y, theta);
grad= 1/m*(X'*error);
%theta = theta - grad;
%J = computeCost(X,y, theta);% unregulated J 
%temp= theta;
%temp(1) =0;
theta(1)=0;
reg = sum(theta.^2)*(lambda/(2*m));
grad = grad+theta*(lambda/m);
J = J+reg;











% =========================================================================

grad = grad(:);

end
