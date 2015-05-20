% Our first step is to plot the data but in order to do so
% we have to reduce it to two dimensions using pca

tic;

clear all; close all; clc;

% First let us load the data
[Xy] = loadData('train.csv');

% % Getting the size of input data
% [m,n] = size(X);

%Full data
X_full = Xy(:,2:end);
X_full = [ones(size(X_full,1),1), X_full];
y_full = Xy(:,1);

% Dividing the data
[Xy_train, Xy_val, Xy_test] = dividerand(Xy', 0.6,0.2,0.2);
Xy_train = Xy_train';
Xy_val = Xy_val';
Xy_test = Xy_test';

X_train = Xy_train(:,2:end);
X_val = Xy_val(:,2:end);
X_test = Xy_test(:,2:end);
y_train = Xy_train(:,1);
y_val = Xy_val(:,1);
y_test = Xy_test(:,1);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

% Reducing the dimensions of data
%[Z] = my_pca(X,2);

% Now that we have data in 2D, we will visualize it
%plot_data(Z,y);

%% ============ Part 2: Compute Cost and Gradient ============

% Now we will train our system using logistic regression with
% regularisation.

% Adding intercept terms to input matrix
X_train = [ones(size(X_train,1),1) X_train];

% Initializing fitting params
initial_theta = zeros(size(X_train,2),1);

% % Compute and display initial cost and theta
% [cost, grad] = costFuncLogReg(initial_theta, X, y);
% 
% fprintf('Cost at initial theta (zeros): %f\n', cost);
% % fprintf('Gradient at initial theta (zeros): \n');
% % fprintf(' %f \n', grad);
% 
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;


% %% ============= Part 3: Optimizing using fminunc without regularisation  =============
% %  In this exercise, you will use a built-in function (fminunc) to find the
% %  optimal parameters theta.
% 
% %  Set options for fminunc
% options = optimset('GradObj', 'on',  'MaxIter', 400);
% 
% %  Run fminunc to obtain the optimal theta
% %  This function will return theta and the cost 
% [theta, cost] = ...
% 	fmincg(@(t)(costFuncLogReg(t, X, y)), initial_theta, options);
% 
% % Print theta to screen
% % fprintf('Cost at theta found by fmincg: %f\n', cost);
% % fprintf('theta: \n');
% % fprintf(' %f \n', theta);
% 
% fprintf('Training complete................');



%% =========== Part 5: Learning Curve for Logistic Regression =============
% %  Next, you should implement the learningCurve function. 
% %
% %  Write Up Note: Since the model is underfitting the data, we expect to
% %                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
% %
% 
% lambda = 0;
% stepSize = 10;
% 
% [error_train, error_val] = ...
%     learningCurve(X_train, y_train, ...
%                   [ones(size(X_val, 1), 1) X_val], y_val, ...
%                   lambda, stepSize);
%               
% m = size(X_train,1);
% 
% plot(1:stepSize:m, error_train, 1:stepSize:m, error_val);
% title('Learning curve for logistic regression')
% legend('Train', 'Cross Validation')
% xlabel('Number of training examples')
% ylabel('Error')
% %axis([0 13 0 150])
% 
% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:stepSize:m
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(floor(i/stepSize) + 1), error_val(floor(i/stepSize) + 1));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% =========== Part 8: Validation for Selecting Lambda =============
% % %  You will now implement validationCurve to test various values of 
% % %  lambda on a validation set. You will then use this to select the
% % %  "best" lambda value.
% % %
% % 
% [lambda_vec, error_train, error_val] = ...
%     validationCurve(X_train, y_train, [ones(size(X_val, 1), 1) X_val], y_val);
% 
% close all;
% plot(lambda_vec, error_train, lambda_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('lambda');
% ylabel('Error');
% 
% fprintf('lambda\t\tTrain Error\tValidation Error\n');
% for i = 1:length(lambda_vec)
% 	fprintf(' %f\t%f\t%f\n', ...
%             lambda_vec(i), error_train(i), error_val(i));
% end
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% 




% %% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
% %  After you have implemented the kernel, we can now use it to train the 
% %  SVM classifier.
% % 
% fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');
% 
% % SVM Parameters
% C = 1; sigma = 0.1;
% 
% % We set the tolerance and max_passes lower here so that the code will run
% % faster. However, in practice, you will want to run the training to
% % convergence.
% model= svmTrain(X_full, y_full, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;




%% ============= Part 3: Optimizing using fminunc with regularisation  =============
% %  In this exercise, you will use a built-in function (fminunc) to find the
% %  optimal parameters theta.
% 
% % Set regularization parameter lambda to 1 (you should vary this)
% lambda = 10;
% 
% %  Set options for fminunc
% % options = optimset('GradObj', 'on',  'MaxIter', 400);
% 
% %  Run fmincg to obtain the optimal theta
% %  This function will return theta and the cost 
% [theta, cost] = trainLogReg(X_full, y_full, lambda);
% 
% % Print theta to screen
% fprintf('Cost at theta found by fmincg: %f\n', cost);
% % fprintf('theta: \n');
% % fprintf(' %f \n', theta);
% 
% fprintf('Training complete................\n\n');


%% ============== Part 4: Predict and Accuracies ==============
%  Now we will predict the outputs for our test set and find how accurate
%  our system is on the basis of how many correct prediction and cost
%  function J
% 
X_test = csvread('test.csv',1,0);

% getting the size of test data
[m_test, n_test] = size(X_test);

% Adding intercept col in X_test
X_test = [ones(m_test,1), X_test];

% % Predicting outputs for logistic regression
% p_train = predict_output(theta, X_full);
% p_test = predict_output(theta, X_test);

% Predicting outputs for svm
p_train = svmPredict(model, X_full);
p_test = svmPredict(model, X_test);

%Calculating accuracy of our result
%[final_cost , grad] = costFuncLogReg(theta, X_test, y_test);

fprintf('Train Accuracy in percentage : %f \n', mean(double(p_train==y_full))*100);
%fprintf('Test Accuracy in percentage : %f \n', mean(double(p_test==y_test))*100);

%fprintf('Final Cost : %f\n', final_cost);
pause;

TimeSpent = toc;