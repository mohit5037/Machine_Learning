function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % Taking a temp matrix as we have to do simultaneous updation of theta
    temp = theta;
    
    % Looping over each and every value of theta
    for th = 1:length(theta)
         
         % Calculating derivative part of gradient descent function
         F = (X*theta - y).*X(:,th);
         g = @(i) F(i);
         J = (1/m)*sum(g([1:m]));
         
         % Calculating new value of theta according to gradient descent function
         temp(th) = theta(th) - alpha*(J);

    end
    
    % Simultaneously updating theta
    theta = temp;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
