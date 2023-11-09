clear;

%read in hate crimes data
H = readcell('hate_crime_data.csv');

%remove text fields
h = H;
h(:,1:2) = []; %cols
h(1,:) = [];%rows

%normalize the dataset
h = cell2table(h);
h = normalize(h, 'range');

%impute nulls
h = table2array(h);
h = knnimpute(h);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(h,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h(trainId,:);
testData = h(testId,:);


[Network, RMSE] = LerningMLP(trainData, [32 64], .005, 4000);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);