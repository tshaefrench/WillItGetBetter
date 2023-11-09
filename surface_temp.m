clear;

%read in dataset
S = readcell('surface_temp.csv');

%remove columns with text
s = S;
s(1,:) = []; %rows
s(:,[1:2]) = []; %cols

%impute nulls
s = cell2table(s);
s = table2array(s);
s = knnimpute(s);
s = normalize(s, 'range');

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(s,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = s(trainId,:);
testData = s(testId,:);

% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [64 128], 0.005, 7000);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

[RMSE] = TestingMLP(testData, Network)