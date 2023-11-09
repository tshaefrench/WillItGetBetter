clear;

%read in hate crimes data
H = readtable('hate_crimes_longitudinal.csv');

%remove text fields 
h = H;
h(:, 1) = []; %remove index row and income
h(:, 2) = []; %remove index row and income
h_header = h.Properties.VariableNames;
h_labels = h(:,1);  %reserve for join later
h(:,1) = []; %cols removes income column...

%normalize the dataset for MLMVN
h = table2array(h);
h_min = min(h(:));
h_max = max(h(:));
h_res = h;
h = new_norm(h);

%impute nulls
h = knnimpute(h);

%train/test split
%Partiion with 25% data as testing 
hpartition = cvpartition(size(h,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h(trainId,:);
testData = h(testId,:);
size(trainData)

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

actuals_2020 = actual_output;
orig_2020 = desired_output;
filename = "2020_Validity_HateCrimes";
var_to_txt(orig_2020, actuals_2020, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(h, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2020_Predicted'};

%add header back and join pred column, reserve header row.
h_input = array2table(h);
h_pred_2020 = [h_labels h_input];
% h_header1 = table2cell(h_header);
% h_header1 = string(h_header1);
% allVars1 = 1:width(h_pred_2020);
% h_pred_2020 = renamevars(h_pred_2020,allVars1, h_header1);
h_pred_2020.Properties.VariableNames = h_header;
h_pred_2020 = [h_pred_2020 pred_val];
h_pred_2020_header = h_pred_2020.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2021 Year Pred%%%%%%%%%%%%%%%%
h_pred_2020(:,1) = []; %remove labels
h_pred_2020 = table2array(h_pred_2020);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(h_pred_2020,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h_pred_2020(trainId,:);
testData = h_pred_2020(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

actuals_2021 = actual_output;
orig_2021 = desired_output;
filename = "2021_Validity_HateCrimes";
var_to_txt(orig_2021, actuals_2021, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(h_pred_2020, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2021_Predicted'};
h_pred_2020_input = array2table(h_pred_2020);
h_pred_2020_input = [h_labels h_pred_2020_input];
h_pred_2020_input.Properties.VariableNames = h_pred_2020_header;
h_pred_2021 = [h_pred_2020_input pred_val];
h_pred_2021_header = h_pred_2021.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2022 Year Pred%%%%%%%%%%%%%%%%
h_pred_2021(:,1) = []; %remove labels
h_pred_2021 = table2array(h_pred_2021);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(h_pred_2021,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h_pred_2021(trainId,:);
testData = h_pred_2021(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

actuals_2022 = actual_output;
orig_2022 = desired_output;
filename = "2022_Validity_HateCrimes";
var_to_txt(orig_2022, actuals_2022, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(h_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};
h_pred_2021_input = array2table(h_pred_2021);
h_pred_2021_input = [h_labels h_pred_2021_input];
h_pred_2021_input.Properties.VariableNames = h_pred_2021_header;
h_pred_2022 = [h_pred_2021_input pred_val];
h_pred_2022_header = h_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
h_pred_2022(:,1) = []; %remove labels
h_pred_2022 = table2array(h_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(h_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h_pred_2022(trainId,:);
testData = h_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] =Net_learn(trainData, [64 128],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

actuals_2023 = actual_output;
orig_2023 = desired_output;
filename = "2023_Validity_HateCrimes";
var_to_txt(orig_2023, actuals_2023, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(h_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
h_pred_2022_input = array2table(h_pred_2022);
h_pred_2022_input = [h_labels h_pred_2022_input];
h_pred_2022_input.Properties.VariableNames = h_pred_2022_header;
h_pred_2023 = [h_pred_2022_input pred_val];
h_pred_2023_header = h_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
h_pred_2023(:,1) = []; %remove labels
h_pred_2023 = table2array(h_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(h_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = h_pred_2023(trainId,:);
testData = h_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

actuals_2024 = actual_output;
orig_2024 = desired_output;
filename = "2024_Validity_HateCrimes";
var_to_txt(orig_2024, actuals_2024, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(h_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
h_pred_2023_input = array2table(h_pred_2023);
h_pred_2023_input = [h_labels h_pred_2023_input];
h_pred_2023_input.Properties.VariableNames = h_pred_2023_header;
h_pred_2024 = [h_pred_2023_input pred_val];
h_pred_2024_header = h_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_pred_2024(:,1) = [];
h_pred_2024 = table2array(h_pred_2024);
h_preds = new_deNorm(h_pred_2024, h_res);
h_preds = array2table(h_preds);
h_preds = [h_labels h_preds];
h_preds.Properties.VariableNames = h_pred_2024_header;
%writetable(h_preds,'hatecrimes_multiyear_predictions_updated.txt','Delimiter',' ')  
