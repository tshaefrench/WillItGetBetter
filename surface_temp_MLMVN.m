clear;

%read in dataset
S = readtable('surface_temp.csv');

%remove columns with text
s = S;
s(:,1) =[];
s_header = s(1,:); %reserve header row
s(1,:) = []; %rows
s_labels = s(:,1); % reserve row labels, minus header row for rejoin
s(:,1) = []; %cols

%normalize the dataset for MLMVN
s = table2array(s);
s_res = s;
min =min(s(:));
%impute nulls
s = knnimpute(s);
s = ( (s - min(s(:)) ) / ( max(s(:)) - min(s(:)) ) ) .* 3*pi/2 + pi/4;

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(s,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = s(trainId,:);
testData = s(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.01712);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE
actuals_2022 = actual_output;
orig_2022 = desired_output;
filename = "2022_Validity_SurfTemp";
var_to_txt(orig_2022, actuals_2022, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(s, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};

%add header back and join pred column, reserve header row.
s_input = array2table(s);
s_pred_2022 = [s_labels s_input];
s_header = table2cell(s_header);
s_pred_2022_header = string(s_header);
allVars1 = 1:width(s_pred_2022);
s_pred_2022 = renamevars(s_pred_2022,allVars1, s_pred_2022_header);
s_pred_2022 = [s_pred_2022 pred_val];
s_pred_2022_header = s_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
s_pred_2022(:,1) = []; %remove labels
s_pred_2022 = table2array(s_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(s_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = s_pred_2022(trainId,:);
testData = s_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.01712);

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

actuals_2023 = actual_output;
orig_2023 = desired_output;
filename = "2023_Validity_SurfTemp";
var_to_txt(orig_2023, actuals_2023, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(s_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
s_pred_2022_input = array2table(s_pred_2022);
s_pred_2022_input = [s_labels s_pred_2022_input];
s_pred_2022_input.Properties.VariableNames = s_pred_2022_header;
s_pred_2023 = [s_pred_2022_input pred_val];
s_pred_2023_header = s_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
s_pred_2023(:,1) = []; %remove labels
s_pred_2023 = table2array(s_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(s_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = s_pred_2023(trainId,:);
testData = s_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.01712);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
actuals_2024 = actual_output;
orig_2024 = desired_output;
filename = "2024_Validity_HateCrimes";
var_to_txt(orig_2024, actuals_2024, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(s_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
s_pred_2023_input = array2table(s_pred_2023);
s_pred_2023_input = [s_labels s_pred_2023_input];
s_pred_2023_input.Properties.VariableNames = s_pred_2023_header;
s_pred_2024 = [s_pred_2023_input pred_val];
s_pred_2024_header = s_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_pred_2024(:,1) = [];
s_pred_2024 = table2array(s_pred_2024);
s_preds = deNorm(s_pred_2024, s_res);
s_preds = array2table(s_preds);
s_preds = [s_labels s_preds];
s_preds.Properties.VariableNames = s_pred_2024_header;
writetable(s_preds,'surface_temp_multiyear_predictions.txt','Delimiter',' ')  
