function [X] = loadData(fName)
% Loads the data from csv file
% [X,y] = loadData(fName) puts the input data in X and labels in y

% excluding header's row.
X = csvread(fName,2,0);

% % Separating out the input and output
% X = all_data(:,2:end);
% y = all_data(:,1);

end