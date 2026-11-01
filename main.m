clear;clc;
obj = ICLAffine();
lambda_values = [0, 1e-5, 1e-4, 1e-3, 5e-3 ,1e-2, 5e-2, 1e-1];
results = obj.runLambdaSweep(lambda_values, fullfile(pwd, 'results'));