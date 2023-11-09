clear;

R = readtable('refugees_clean.csv');

r = R;
r(1,:) =[]; %drop row with indices labelled
r_header = r(1,:);
r(1,:) = []; %rows
r_labels = r(:,1);
r(:,1) = []; %cols

%normalize the dataset for MLMVN
r = table2array(r);
r_res = r;
r = ( (r - min(r(:)) ) / ( max(r(:)) - min(r(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
r = knnimpute(r);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(r,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = r(trainId,:);
testData = r(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.0036518);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(r, hidneur_weights1, hidneur_weights2, outneur_w);
%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};


r_pred_2023_input = array2table(r);
r_pred_2023 = [r_labels r_pred_2023_input];
%u_pred_2023_input.Properties.VariableNames = u_header;
r_header1 = table2cell(r_header);
r_header1 = string(r_header1);
allVars1 = 1:width(r_pred_2023);
r_pred_2023 = renamevars(r_pred_2023,allVars1, r_header1);
r_pred_2023 = [r_pred_2023 pred_val];
r_pred_2023_header = r_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
r_pred_2023(:,1) = []; %remove labels
r_pred_2023 = table2array(r_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(r_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = r_pred_2023(trainId,:);
testData = r_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.0036518);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(r_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);
%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};

%add back to table
r_pred_2023_input = array2table(r_pred_2023);
r_pred_2023_input = [r_labels r_pred_2023_input];
r_pred_2023_input.Properties.VariableNames = r_pred_2023_header;
r_pred_2024 = [r_pred_2023_input pred_val];
r_pred_2024_header = r_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

r_pred_2024(:,1) = [];
r_pred_2024 = table2array(r_pred_2024);
r_preds = new_deNorm(r_pred_2024, r_res);
r_preds = array2table(r_preds);
r_preds = [r_labels r_preds];
r_preds.Properties.VariableNames = r_pred_2024_header;
writetable(r_preds,'refugee_multiyear_predictions_updated.txt','Delimiter',' ')  
