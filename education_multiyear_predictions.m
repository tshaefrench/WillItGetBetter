clear;

%read in education attainment data
E = readtable('education_data.csv');

%remove first column and row because text data
e = E;
e_header = e(1,:); %reserve header for join later
e(1,:) = []; %rows
e_labels = e(:,1); %reserve for join later minus header row
e(:,1) = [];

%normalize data
%normalize the dataset for neural network
e = table2array(e);
e_res = e;
e = ( (e - min(e(:)) ) / ( max(e(:)) - min(e(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
e = knnimpute(e);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(e,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e(trainId,:);
testData = e(testId,:);
numinput = size(trainDat)

%%%%%%Train and Test%%%%%%%%%%

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2018_Predicted'};

%add header back and join pred column, reserve header row.
e_input = array2table(e);
e_pred_2018 = [e_labels e_input];
e_header1 = table2cell(e_header);
e_header1 = string(e_header1);
allVars1 = 1:width(e_pred_2018);
e_pred_2018 = renamevars(e_pred_2018,allVars1, e_header1);
e_pred_2018 = [e_pred_2018 pred_val];
e_pred_2018_header = e_pred_2018.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2019 Year Pred%%%%%%%%%%%%%%%%
e_pred_2018(:,1) = []; %remove labels
e_pred_2018 = table2array(e_pred_2018);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2018,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2018(trainId,:);
testData = e_pred_2018(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2018, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2019_Predicted'};
e_pred_2018_input = array2table(e_pred_2018);
e_pred_2018_input = [e_labels e_pred_2018_input];
e_pred_2018_input.Properties.VariableNames = e_pred_2018_header;
e_pred_2019 = [e_pred_2018_input pred_val];
e_pred_2019_header = e_pred_2019.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2020 Year Pred%%%%%%%%%%%%%%%%
e_pred_2019(:,1) = []; %remove labels
e_pred_2019 = table2array(e_pred_2019);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2019,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2019(trainId,:);
testData = e_pred_2019(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2019, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2020_Predicted'};
e_pred_2019_input = array2table(e_pred_2019);
e_pred_2019_input = [e_labels e_pred_2019_input];
e_pred_2019_input.Properties.VariableNames = e_pred_2019_header;
e_pred_2020 = [e_pred_2019_input pred_val];
e_pred_2020_header = e_pred_2020.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2021 Year Pred%%%%%%%%%%%%%%%%
e_pred_2020(:,1) = []; %remove labels
e_pred_2020 = table2array(e_pred_2020);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2020,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2020(trainId,:);
testData = e_pred_2020(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.018289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2020, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2021_Predicted'};
e_pred_2020_input = array2table(e_pred_2020);
e_pred_2020_input = [e_labels e_pred_2020_input];
e_pred_2020_input.Properties.VariableNames = e_pred_2020_header;
e_pred_2021 = [e_pred_2020_input pred_val];
e_pred_2021_header = e_pred_2021.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2022 Year Pred%%%%%%%%%%%%%%%%
e_pred_2021(:,1) = []; %remove labels
e_pred_2021 = table2array(e_pred_2021);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2021,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2021(trainId,:);
testData = e_pred_2021(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};
e_pred_2021_input = array2table(e_pred_2021);
e_pred_2021_input = [e_labels e_pred_2021_input];
e_pred_2021_input.Properties.VariableNames = e_pred_2021_header;
e_pred_2022 = [e_pred_2021_input pred_val];
e_pred_2022_header = e_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
e_pred_2022(:,1) = []; %remove labels
e_pred_2022 = table2array(e_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2022(trainId,:);
testData = e_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
e_pred_2022_input = array2table(e_pred_2022);
e_pred_2022_input = [e_labels e_pred_2022_input];
e_pred_2022_input.Properties.VariableNames = e_pred_2022_header;
e_pred_2023 = [e_pred_2022_input pred_val];
e_pred_2023_header = e_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
e_pred_2023(:,1) = []; %remove labels
e_pred_2023 = table2array(e_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(e_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e_pred_2023(trainId,:);
testData = e_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.18289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
e_pred_2023_input = array2table(e_pred_2023);
e_pred_2023_input = [e_labels e_pred_2023_input];
e_pred_2023_input.Properties.VariableNames = e_pred_2023_header;
e_pred_2024 = [e_pred_2023_input pred_val];
e_pred_2024_header = e_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gets added somewhere in 2023??
e_pred_2024(:,1) = [];
e_pred_2024 = table2array(e_pred_2024);
e_preds = new_deNorm(e_pred_2024, e_res);
e_preds = array2table(e_preds);
e_preds = [e_labels e_preds];
e_preds.Properties.VariableNames = e_pred_2024_header;
writetable(e_preds,'education_multiyear_predictions_updated.txt','Delimiter',' ')  
