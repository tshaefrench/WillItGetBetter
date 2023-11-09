clear;

%read in dataset
P = readcell('premature_mortality_clean.csv');

%remove columns with text
p = P;
p(:,1) = []; %rows
p(1,:) = [];%cols

%normalize dataset
p = cell2table(p);
p = normalize(p, 'range');

%impute nulls
p = table2array(p);
p = knnimpute(p);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(p,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p(trainId,:);
testData = p(testId,:);

[Network, RMSE] = LerningMLP(trainData, [8 16], 0.005, 7125);
[RMSE] = TestingMLP(testData, Network);
