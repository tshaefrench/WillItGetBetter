clear;

%read in dataset
V = readcell('violent_crime_clean.csv');

%remove columns with text
v = V;
v(1,:) = []; %rows
v(:,1) = []; %cols

%impute nulls
v = cell2table(v);
v = table2array(v);
v = knnimpute(v);
v = normalize(v, 'range');

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(v,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v(trainId,:);
testData = v(testId,:);


% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [16 32], 0.005, 7000 );
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

 [RMSE] = TestingMLP(testData, Network)