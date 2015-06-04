clear all; close all; clc;

n=4; % *Line 1*
pm=12; ps=10; % population mean and std *Line 2*
figure; hold on; % start an empty figure *Line 3*
%*Line 4*
miss_count = 0;
for i=1:1000 % *Line 5*
x=pm+ps*randn(n,1); % *Line 6*
mx=mean(x); % *Line 7*
zcr=1.96; % *Line 8*
sem=ps/sqrt(n); % standard error of the mean *Line 9*
me=zcr*sem; % margin of error, *Line 10*
CI1= mx-me; % lower CI bound *Line 11*
CI2= mx+me; % upper CI bound *Line 12*

plot(i*ones(1,n), x, 'k.','markersize',5); % show points *Line 13*
plot(i*ones(1,2),[CI1, CI2],'g+') % show CI * Line 14*
if pm<CI1 || pm>CI2 % missed *Line 15*
plot(i,pm,'r*','markersize',10) %*Line 16*
miss_count = miss_count + 1;
% *Line 17*
end % *Line 18*
%*Line 19*
end
xlabel('experiment number') % *Line 20*
ylabel('observations ') % *Line 21*
title([num2str(miss_count),' misses out of 1000']) % Line 22 %*Line 22 *

