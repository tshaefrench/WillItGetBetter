clear;

%read in dataset
G = readcell('Orientation_Gender.csv');

%remove columns with text
g = G;
g([1,2],:) = []; %rows
g(:,1:4) = []; %cols
g(:, 15) = []; %text col Predom. Rel
g(:, 14) = []; %text col Democracy

%normalize dataset
g = cell2table(g);
g = normalize(g, 'range');

%impute nulls
g = table2array(g);
g = knnimpute(g);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(g,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = g(trainId,:);
testData = g(testId,:);

% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [8 16], 0.005, 8000);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network)