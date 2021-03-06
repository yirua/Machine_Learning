%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data_stock.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X_norm mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X_norm];
fprintf('First 10 examples from the dataset after add one intercept term: \n');
fprintf(' x = [%.2f %.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.009;
%alpha = 0.05;
%alpha = 0.1;
%alpha = 0.2;
%alpha = 2;
num_iters = 1400;

% Init Theta and Run Gradient Descent 
theta = zeros(5, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the index of a features= [-297,-45, 0.54, 1216]
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
training_data = load('ex1_multi_stock_testing.txt');
features =training_data(:, 1:4);

%features = [-278,48, 1.49, 1061;-160,-44,1.32, 1090;-278,-44,0.35,796;494,48,0.77,92;309,115,3.99,831;195,38,1.57,1093;321,-4,1.26,1046;-186,-47,2.35,784.93;
% 463,-22,0.9,844.7;150,-80,-2.08,935.81];
%create a vector of the prediction features, 2016.07.04--07.08, last four is from 2017-02".
%actual_index=[2988;3054.21;3012;3159.17;3196.61;3202.08;3253.43;3212;3237.45;3269]
actual_index= training_data(:, 5);
%features_1 = [585,-97,1.27, 263]; % 2016.11.28--12.2
%features_2 = [321,42,1.37, 1343]; % 2016.5.3--5.6
% features = [-278,48, 1.49, 1061] ;% 2016-7.04--08
%features_1 = [-160,-44,1.32, 1090]; % 2016.7.11--7.15
%features_2 = [-278,-44,0.35,796]; % 2016.7.18--7.22
data_num = size(features,1);
features = features-mu;
features = features ./sigma;
%for i=1:data_num,
% features(i, :) = features(i, :)-mu %subtract 'mu',
% features(i, :) = features(i, :) ./ sigma %then divide element-wise by 'sigma'. mu and sigma were returned by featureNormalize().
%end;
pause;
% features(:,1) = ones(size(features,2)) %add a bias unit
features = [ones(rows(features), 1) features];
%X = [ones(m, 1) X];
%b = [ones(size(b, 1), 1) b];
%features_1= [ones(rows(features_1),1) features_1];
%multiply by theta
price = zeros(rows(features),1);
%for i=1:data_num,
%	price(i) = features(i, :)*theta;
	%price = sum(features(i, :)); 
%end;
price = features * theta;  
%%%%%%%%%%%%%%%%%%%%5


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
% ============================================================
fprintf('comparison with gradient descent prediction.\n');
for j=1:length(price),
fprintf(['%d: Actual shanghai index by volumn 2016.7.08--2017.3' ...
         '(using gradient descent):\n $%f\n'], j,actual_index(j, :));
fprintf(['Predicted shanghai index by volumn 2016.11.14--2017.3' ...
         '(using gradient descent):\n $%f\n'], price(j, :));
fprintf(['Actual shanghai index by volumn 2016.7.08--2017.3' ...
         '(using gradient descent) minus Predicted one:\n $%f\n'], actual_index(j, :)-price(j, :));
end;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data_stock.txt');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
%fprintf('the theta vector after normal equation..\n');
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the shanghai index with data from sipf.com
% ====================== YOUR CODE HERE ======================

%features = [-278,48, 1.49, 1061;-160,-44,1.32, 1090;-278,-44,0.35,796;494,48,0.77,92;309,115,3.99,831;195,38,1.57,1093;321,-4,1.26,1046;-186,-47,2.35,784.93;463,-22,0.9,844.7;150,-80,-2.08,935.81];
features =training_data(:, 1:4);
% matrix of input data for predictions


%add a bias unit
 features= [ones(rows(features),1) features];

%multiply by theta
price = zeros(rows(features),1);
%for i=1:rows(features),
%	price(i) =features(i, :) *theta;
	price= (features *theta); % m*n x n*1 -> m*1
%end;
%features = features*theta;
%features_1 = features_1*theta;

% ============================================================

for j=1:length(price),
fprintf(['Actual shanghai index by volumn 2016.7.08--2017.2' ...
         '(using normal equation):\n $%f\n'], actual_index(j, :));
fprintf(['Predicted shanghai index by volumn 2016.11.14--11.18' ...
         '(using normal equation):\n $%f\n'], price(j, :));
fprintf(['%d: Diff of Predicted shanghai index by volumn 2016.11.14--11.18' ...
         '(using normal equation) between actual_index:\n $%f\n'],j,actual_index(j, :)-price(j, :));
end;
plotStockData(features(:,4), price,actual_index);%plot(x, y, 'rx', 'MarkerSize', 10);% Plot the data
hold on;

%fprintf(['Predicted price of a 2016.11.28--12.2 ' ...
%         '(using normal equations):\n $%f\n'], price_1);
%fprintf(['Predicted price of a 2016.5.3--5.6 ' ...
%         '(using normal equations):\n $%f\n'], price_2);

