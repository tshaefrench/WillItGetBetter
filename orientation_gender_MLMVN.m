clear;

%read in dataset
G = readmatrix('Orientation_Gender.csv');

%remove columns with text
g = G;
g([1,2],:) = []; %rows
g(:,1:4) = []; %cols
g(:, 15) = []; %text col Predom. Rel
g(:, 14) = []; %text col Democracy
g(:, 3) = []; %GDP col factorized

%normalize the dataset for MLMVN
g = ( (g - min(g(:)) ) / ( max(g(:)) - min(g(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
g = knnimpute(g);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(g,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = g(trainId,:);
testData = g(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.018289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE
