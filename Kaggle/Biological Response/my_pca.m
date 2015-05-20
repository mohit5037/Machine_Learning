function [Z] = my_pca(X,k)
% Compute the reduced k dim matrix Z of input vector X
% X = inputs
% k = final reduced dimension

% Calculating size of X
[m,n] = size(X);

% Fixing the size of Z
Z = zeros(size(X,1),k);

% Mean normalisation and feature scaling
mean_matrix = ones(m,1) * mean(X);
std_matrix = ones(m,1) * std(X);

X = (X - mean_matrix);

% Calculating covariance matrix
sigma = (1/m)*(X'*X);

% Computing eigen vectors etc
[U,S,V] = svd(sigma);

% Computing Z by reducing the matrix U
Z = X*U(:,1:k);

end
