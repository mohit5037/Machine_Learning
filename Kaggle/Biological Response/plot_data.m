function plot_data(X,y)
%  Plots the data in 2D form
% X = input matrix
% y = output labels

y_pos = find(y == 1);
y_neg = find(y == 0);

figure;
hold on;

% Plotting positive points
plot(X(y_pos,1), X(y_pos,2), 'k+','LineWidth', 2, 'MarkerSize' , 7);

% Plotting negative points
plot(X(y_neg,1), X(y_neg,2), 'ro','LineWidth', 2, 'MarkerSize' , 7);

hold off;

end