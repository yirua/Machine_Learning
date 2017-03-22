function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_vec= [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
%C_vec= [0.1, 0.3, 1, 3, 10, 30]';
%sigma_vec= [0.1, 0.3, 1, 3, 10, 30]';
%C_vec=[1]
%sigma_vec = [0.3];
m= length(C_vec);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
x1 = [1 2 1]; x2 = [0 4 -1];
svm_errors= ones(m^2, 3);
k=1;
for i=1:m,
	fprintf('temp_C:\n');
	temp_C= C_vec(i);
	
	for j=1:m,
		fprintf('temp_sigma:\n');
		temp_sigma=sigma_vec(j);
		model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma));
		predictions = svmPredict(model, Xval);
		prediction_error= mean(double(predictions ~= yval));
		fprintf('The svm_errors are: \n');		
		svm_errors(k,:)=[temp_C,temp_sigma,prediction_error];% set the matrix value with temp_C, temp_sigma and prediction_error..
		k=k+1; % try to record the number of svm_errors
		
	end
end
fprintf('svm_errors:\n');
svm_errors;
fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('show the value and index of minimum:\n')
[value, min_index]=min(svm_errors) % get the min value in the matrix
C=svm_errors(min_index(3),1);
sigma=svm_errors(min_index(3),2);
fprintf('Program paused. Press enter to continue.\n');
pause;
% =========================================================================

end
