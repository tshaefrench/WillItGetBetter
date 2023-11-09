clear;
%read in hate crimes data
Hg = readcell('housing_cost.csv');

%remove text fields
hg = Hg;
hg(:,1:11) = [];
hg(1,:) = [];

%normalize the dataset
hg = cell2table(hg);
hg = normalize(hg, 'range');

%impute nulls
hg = table2array(hg);
hg = knnimpute(hg);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(hg,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = hg(trainId,:);
testData = hg(testId,:);

[Network, RMSE] = LerningMLP(trainData, [16 32], .0001, 3500);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);