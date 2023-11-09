clear;

%read in dataset as cell for later processing
W = readcell('women_parliament.csv');

%drop first column with text
w = W;
w(:,1:2) = [];
w(1:2,:) = [];

%normalize data between 0 and 1
w = cell2table(w);
w = normalize(w, 'range');

%impute nulls
w = table2array(w);
w = knnimpute(w);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(w,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = w(trainId,:);
testData = w(testId,:);


% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [8 16], 0.005, 6500 );
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

 [RMSE] = TestingMLP(testData, Network)