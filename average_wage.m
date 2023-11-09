clear;

%read in average wage dataset
A = readcell('avg_wage.csv');

%reserve the text data for joining later. drop header col & row
a =A;
a(:,1) = [];
a(1,:) = [];

%normalize the dataset for neural network
%n(cellfun(@isempty,n)) = {"NaN"};
a = cell2table(a);
a = normalize(a, 'range');

%impute nulls with nearest neighbor method
a = table2array(a);
a = knnimpute(a);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(a,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = a(trainId,:);
testData = a(testId,:);


[Network, RMSE] = LerningMLP(trainData, [8 16], .005, 7250);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);