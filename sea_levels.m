clear;

%read in dataset
C = readcell('sea_levels.csv');

%remove columns with text
c = C;
c(1,:) = []; %rows

%address datetime data
c = cell2table(c);
c.c1 = datetime(c.c1,'Format', 'dd-MM-yyyy');

cres = c.c1;

c(:,1) = []; %first col for knnimpute

%impute nulls
c = table2array(c);
c = knnimpute(c);
c = array2table(c);
cres = array2table(cres);
c = [c, cres];
c = normalize(c, 'range');

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(c,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = c(trainId,:);
testData = c(testId,:);

% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(trainData, [8 64], 0.005, 6000 );
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

 [RMSE] = TestingMLP(testData, Network)