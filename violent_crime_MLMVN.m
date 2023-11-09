clear;

%read in dataset
V = readtable('violent_crime_clean.csv');

%remove columns with text
v = V;
v_header = v(1,:); %header reserve
v(1,:) = []; %rows
v_labels = v(:,1);
v(:,1) = []; %cols

%normalize the dataset for MLMVN
v = table2array(v);
v_res = v;
v = ( (v - min(v(:)) ) / ( max(v(:)) - min(v(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
v = knnimpute(v);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(v,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v(trainId,:);
testData = v(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.00135);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(v, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2020_Predicted'};

%add header back and join pred column, reserve header row.
v_input = array2table(v);
v_pred_2020 = [v_labels v_input];
v_header1 = table2cell(v_header);
v_header1 = string(v_header1);
allVars1 = 1:width(v_pred_2020);
v_pred_2020 = renamevars(v_pred_2020,allVars1, v_header1);
v_pred_2020.Properties.VariableNames = v_header1;
v_pred_2020 = [v_pred_2020 pred_val];
v_pred_2020_header = v_pred_2020.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2021 Year Pred%%%%%%%%%%%%%%%%
v_pred_2020(:,1) = []; %remove labels
v_pred_2020 = table2array(v_pred_2020);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(v_pred_2020,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v_pred_2020(trainId,:);
testData = v_pred_2020(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.00135);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(v_pred_2020, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2021_Predicted'};
v_pred_2020_input = array2table(v_pred_2020);
v_pred_2020_input = [v_labels v_pred_2020_input];
v_pred_2020_input.Properties.VariableNames = v_pred_2020_header;
v_pred_2021 = [v_pred_2020_input pred_val];
v_pred_2021_header = v_pred_2021.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2022 Year Pred%%%%%%%%%%%%%%%%
v_pred_2021(:,1) = []; %remove labels
v_pred_2021 = table2array(v_pred_2021);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(v_pred_2021,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v_pred_2021(trainId,:);
testData = v_pred_2021(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.00135);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(v_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};
v_pred_2021_input = array2table(v_pred_2021);
v_pred_2021_input = [v_labels v_pred_2021_input];
v_pred_2021_input.Properties.VariableNames = v_pred_2021_header;
v_pred_2022 = [v_pred_2021_input pred_val];
v_pred_2022_header = v_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
v_pred_2022(:,1) = []; %remove labels
v_pred_2022 = table2array(v_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(v_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v_pred_2022(trainId,:);
testData = v_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] =Net_learn(trainData, [16 32],0.00135);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(v_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
v_pred_2022_input = array2table(v_pred_2022);
v_pred_2022_input = [v_labels v_pred_2022_input];
v_pred_2022_input.Properties.VariableNames = v_pred_2022_header;
v_pred_2023 = [v_pred_2022_input pred_val];
v_pred_2023_header = v_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
v_pred_2023(:,1) = []; %remove labels
v_pred_2023 = table2array(v_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(v_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = v_pred_2023(trainId,:);
testData = v_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.00135); 
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(v_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
v_pred_2023_input = array2table(v_pred_2023);
v_pred_2023_input = [v_labels v_pred_2023_input];
v_pred_2023_input.Properties.VariableNames = v_pred_2023_header;
v_pred_2024 = [v_pred_2023_input pred_val];
v_pred_2024_header = v_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_pred_2024(:,1) = [];
v_pred_2024 = table2array(v_pred_2024);
v_preds = new_deNorm(v_pred_2024, v_res);
v_preds = array2table(v_preds);
v_preds = [v_labels v_preds];
v_preds.Properties.VariableNames = v_pred_2024_header;
writetable(v_preds,'violent_crimes_multiyear_predictions_updated.txt','Delimiter',' ')  
