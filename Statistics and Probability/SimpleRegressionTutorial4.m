clear all; close all; clc;

n=50;
x=linspace(0,1,n); % linearly spaced vector a length n
y= 10*x + 3 + randn(1,n);
mx=mean(x), my=mean(y), sx=std(x,1);
c= mean((x-mx).*(y-my)) %covariance
bh1=c/sx^2
bh0=my-bh1*mx
yhat=bh0+bh1*x; %regression line
figure;
plot(x,y,'.')
hold on
plot(x,yhat,'r')
xlabel('x'), ylabel('y')
title(['Regression yhat=',num2str(bh1),'*x+',num2str(bh0)])

x=x(:); % make x a column
y=y(:); % make y a column
XX=[ones(n,1),x]; % create the design matrix
bh=(XX'*XX)^-1*XX'*y % OLS estimate of b

yhat=XX*bh;
hold on
plot(x,yhat,'g+','linewidth',2)

e=y-yhat; % residuals
dof= n-rank(XX); % degrees of freedom
vn=sum(e.^2)/dof % estimated noise variance