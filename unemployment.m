clear;

%read in dataset
U = readcell('unemployment.csv');

%remove columns with text
u = U;
u(1,:) = []; %rows
u(:,[1, 12]) = []; %cols

%impute nulls
u = cell2table(u);
u = table2array(u);
u = knnimpute(u);
u = normalize(u, 'range');

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(u,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = u(trainId,:);
testData = u(testId,:);


% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [8 16], 0.005, 6000);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

 [RMSE] = TestingMLP(testData, Network)