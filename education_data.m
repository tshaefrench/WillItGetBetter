clear;

%read in education attainment data
E = readcell('education_data.csv');

%remove first column and row because text data
e = E;
e(:,1) = [];
e(1,:) = [];

%normalize data
e = cell2table(e);
e = normalize(e, 'range');

%impute nulls
e = table2array(e);
e = knnimpute(e);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(e,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e(trainId,:);
testData = e(testId,:);


%%%%%%Train and Test%%%%%%%%%%
[Network, RMSE] = LerningMLP(trainData, [8 16], 0.005, 8000);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);





































































