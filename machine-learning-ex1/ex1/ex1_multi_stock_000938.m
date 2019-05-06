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
choice =0; % 0 means with the VolDol, 1 means NoVolDol
%choice =1;
if (choice==0),
  data = load('shanghai_index_000938_training.csv');
  X = data(:, (2:8));
  y = data(:, 10);
else  
  data = load('shanghai_index_000938_training_NoVolDol.csv');
  X = data(:, (2:6));
  y = data(:, 8);
end;
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
if(choice==0),
  fprintf(' x = [ %.2f %.2f %.2f %.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');
else
  fprintf(' x = [ %.2f %.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');
end;  
fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X_norm mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X_norm];
fprintf('First 10 examples from the dataset after normalization: \n');
if(choice==0),
  fprintf(' x = [%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');
else %NoVolDol, the volumns are missing one
  fprintf(' x = [ %.2f %.2f %.2f %.2f %.2f %.2f], y = %.2f \n', [X(1:10,:) y(1:10,:)]');
end;
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
alpha = 0.005;
%alpha = 0.05;
%alpha = 0.1;
%alpha = 0.2;
%alpha = 2;
num_iters = 1400;

% Init Theta and Run Gradient Descent 
if (choice==0),
  theta = zeros(8, 1);
else
  theta = zeros(6,1);
end;
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the index of a features= [-297,-45, 0.54, 1216]
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
if (choice==0),
  testing_data = load('shanghai_index_000938_testing_2019.csv');
  features =testing_data(:, 2:8);
  actual_index= testing_data(:, 10);
else
  testing_data = load('shanghai_index_000938_testing_2019_NoVolDol.csv');
  features =testing_data(:, 2:6);
  actual_index= testing_data(:, 8);
end;


data_num = size(features,1);
features = features-mu;
features = features ./sigma;
fprintf('First 10 features without bias from the dataset: \n');
if (choice==0),
fprintf(' features = [%.2f %.2f %.2f %.2f %.2f %.2f %.2f ] \n', [features(1:10,:) ]);
else
fprintf(' features = [%.2f %.2f %.2f %.2f %.2f] \n', [features(1:10,:) ]);
end;
%for i=1:data_num,
% features(i, :) = features(i, :)-mu %subtract 'mu',
% features(i, :) = features(i, :) ./ sigma %then divide element-wise by 'sigma'. mu and sigma were returned by featureNormalize().
%end;
fprintf('Program paused. Press enter to continue.\n');
pause;
%features(:,1) = ones(size(features,1)) %add a bias unit
features_bias = [ones(data_num,1), features];
% Add intercept term to X
% X = [ones(m, 1) X_norm];
%features(:,1) = ones(size(features,1));
fprintf('First 10 features with bias from the dataset: \n');
if (choice==0),
fprintf(' features = [%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f] \n', [features_bias(1:10,:)] );
else
  fprintf(' features = [%.2f %.2f %.2f %.2f %.2f %.2f ] \n', [features(1:10,:) ]);
 end;
%X = [ones(m, 1) X];
%b = [ones(size(b, 1), 1) b];
%features_1= [ones(rows(features_1),1) features_1];
%multiply by theta

price = zeros(rows(features_bias),1);
%for i=1:data_num,
%	price(i) = features(i, :)*theta;
	%price = sum(features(i, :)); 
%end;
price = features_bias * theta;  
%%%%%%%%%%%%%%%%%%%%5


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 
% ============================================================
fprintf('comparison with gradient descent prediction.\n');
for j=1:length(price),
fprintf(['%d: Actual 000938 price starting 2019' ...
         '(using gradient descent):\n $%f\n'], j,actual_index(j, :));
fprintf(['Predicted 000938 price starting 2019' ...
         '(using gradient descent):\n $%f\n'], price(j, :));
fprintf(['Error in Actual 000938 price starting 2019' ...
         '(using gradient descent) compare with Predicted one:\n $%f\n'], (actual_index(j, :)-price(j, :))/price(j,1));
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
if (choice==0),
  data = load('shanghai_index_000938_training.csv');
  X = data(:, 2:8);
  y = data(:, 10);
else
  data = load('shanghai_index_000938_training_NoVolDol.csv');
  X = data(:, 2:6);
  y = data(:, 8);
end;  
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
if (choice==0),
  features =testing_data(:, 2:8);
  actual_index= testing_data(:, 10);
else
  features =testing_data(:, 2:6);
  actual_index= testing_data(:, 8);
end,
m = length(actual_index);
% matrix of input data for predictions

%features_bias = [ones(data_num,1), features];
%add a bias unit
 features_bias= [ones(m,1) features];

%multiply by theta
price = zeros(rows(features_bias),1);
%for i=1:rows(features),
%	price(i) =features(i, :) *theta;
	price= (features_bias *theta); % m*n x n*1 -> m*1
%end;


% ============================================================

for j=1:length(price),
fprintf(['Actual 000938 price starting 2019' ...
         '(using normal equation):\n $%f\n'], actual_index(j, :));
fprintf(['Predicted 000938 price starting 2019 '...
         '(using normal equation):\n $%f\n'], price(j, :));
fprintf(['%d: Diff of Predicted 000938 in 2019-18' ...
         '(using normal equation) between actual_index:\n $%f\n'],j,(actual_index(j, :)-price(j, :))/price(j,:));
end;
plotStockData(features_bias(:,7), price,actual_index);%plot(x, y, 'rx', 'MarkerSize', 10);% Plot the data
xlabel('number of samples');
ylabel('stock price');
hold on;



