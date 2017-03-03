function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%    theta_0= theta(1,1);
%	temp =0;
%        temp_1 =0;
%	for i=1:m,
% 		temp= temp +(theta(1,1)+theta(2,1)*X(i,2)-y(i,1)); % sum of variance
		% sum of variance *X(i,2)		
%		temp_1 = temp_1+((theta(1,1)+theta(2,1)*X(i,2)-y(i,1))*X(i,2)); 

%	endfor

%	temp = temp/m;
%        temp_1 = temp_1/m;
%    theta_0 = theta_0 -(alpha*temp);
%    theta_1 = theta(2,1);
%    theta_1 = theta_1 -(alpha*temp_1);
%    theta(1,1) = theta_0;
%    theta(2,1) = theta_1;

	h = X*theta;
	error = h-y;

	gradient_error= alpha/m*(X'*error);
	theta = theta - gradient_error;
  

% print theta to screen
%	fprintf('Theta found by gradient descent: ');
%	fprintf('%f %f \n', theta(1), theta(2));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	J_history(iter);

end

end
