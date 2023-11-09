
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%This Works!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

%read in inflation dataset from worldbank.org delete null lines of data.
N = readcell('inflation_clean.csv');
%rows_to_delete = find(all(cellfun(@ismissing,N(1:end,5:end)),2));

%N(rows_to_delete,:) = [];

%drop cols 2-4. All text data and headers.
N([1,2],:) = [];
N(:,[2,3,4]) = [];

%reserve the text data for joining later. drop header col & row
n = N;
n(:,1) = [];
n(1,:) = [];

%normalize the dataset for neural network
%n(cellfun(@isempty,n)) = {"NaN"};
n = cell2table(n);
n = normalize(n, 'range');

%impute nulls with nearest neighbor method
n = table2array(n);
n = knnimpute(n);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(n,1),'Holdout',0.3); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = n(trainId,:);
testData = n(testId,:);


%%%%MLP%%%%%%%%
[Network, RMSE] = LerningMLP(trainData, [8 16], .005, 7500);
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network
[RMSE] = TestingMLP(testData, Network);
