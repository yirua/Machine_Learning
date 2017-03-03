function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1)
n = size(X, 2)
num_labels = size(Theta2, 1)

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
%for i=1:m,
%	if (sigmoid(X(i, :)*Theta1') >=0.5)
%	 p(i,1) =1;
%	else 
%	 p(i,1) =0;   
%	endif
%end
X = [ones(m, 1) X];


p1= sigmoid(X*Theta1');

[v,p]= max(p1,[],2);
p1= [ones(m, 1) p1];
p2 = sigmoid(p1*Theta2');

[v,p] = max(p2, [], 2);






% =========================================================================


end
