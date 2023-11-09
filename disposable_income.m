clear;

%read in disposable income dataset
D = readcell('disposable_income.csv');

%reserve the text data for joining later. drop header col & row
d =D;
d(:,1) = [];
d(1,:) = [];

%normalize the dataset for neural network
%n(cellfun(@isempty,n)) = {"NaN"};
d = cell2table(d);
d = normalize(d, 'range');

%impute nulls with nearest neighbor method
d = table2array(d);
d = knnimpute(d);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(d,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = d(trainId,:);
testData = d(testId,:);

%%%%%%Train and Test%%%%%%%%%%
[Network, RMSE] = LerningMLP(trainData, [32 128], 0.005, 9500);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);