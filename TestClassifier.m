clear all
close all
clc
format compact

load MNIST
tic;
labelProb = ClassifyMNIST(data);
toc;
[~, digitTrue] = max(label,[],2);
digitTrue = digitTrue-1;
[~, digitPredicted] = max(labelProb,[],2);
digitPredicted = digitPredicted-1;
correct = mean(digitTrue==digitPredicted);
fprintf('MNIST classified with %.2f%% correct\n', correct*100);