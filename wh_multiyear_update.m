clear;

%read in dataset
WH = readtable('world_happiness_longitudinal.csv');

%remove columns with text
wh = WH;
wh(:,1) = []; %incorrect col
wh_header = wh.Properties.VariableNames; %get header row from vars. 
wh_labels = wh(:,1);
wh(:,1) = []; %cols

%normalize the dataset for MLMVN
wh = table2array(wh);
wh_min = min(wh(:));
wh_max = max(wh(:));
%wh = ( (wh - min(wh(:)) ) / ( max(wh(:)) - min(wh(:)) ) ) .* 3*pi/2 + pi/4;
wh = neg_norm(wh);

%impute nulls
wh = knnimpute(wh);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(wh,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = wh(trainId,:);
testData = wh(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.0014838125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(wh, hidneur_weights1, hidneur_weights2, outneur_w);

% pred_val = array2table(pred_val);
% pred_val.Properties.VariableNames = {'2023_Predicted'};

%add header back and join pred column, reserve header row.
% wh_input = array2table(wh);
% v_pred_2020 = [v_labels wh_input];
% v_header1 = table2cell(v_header);
% v_header1 = string(v_header1);
% allVars1 = 1:width(v_pred_2020);
% v_pred_2020 = renamevars(v_pred_2020,allVars1, v_header1);
% v_pred_2020.Properties.VariableNames = v_header1;
% v_pred_2020 = [v_pred_2020 pred_val];
% v_pred_2020_header = v_pred_2020.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
wh_input = array2table(wh);
wh_input = [wh_labels wh_input];
% %wh_header1 = table2cell(wh_header);
% wh_header1 = string(wh_header);
% allVars1 = 1:width(wh_input);
% wh_pred_2023 = renamevars(wh_input,allVars1, wh_header1);
wh_pred_2023.Properties.VariableNames = wh_header;
wh.Properties.VariableNames = wh_header;
wh_pred_2023 = [wh_input pred_val];
wh_pred_2023_header = wh_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
wh_pred_2023(:,1) = []; %remove labels
wh_pred_2023 = table2array(wh_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(wh_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = wh_pred_2023(trainId,:);
testData = wh_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.0014838125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(wh_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
wh_pred_2023_input = array2table(wh_pred_2023);
wh_pred_2023_input = [wh_labels wh_pred_2023_input];
wh_pred_2023_input.Properties.VariableNames = wh_pred_2023_header;
wh_pred_2024 = [wh_pred_2023_input pred_val];
wh_pred_2024_header = wh_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

wh_pred_2024(:,1) = [];
wh_pred_2024 = table2array(wh_pred_2024);
wh_preds = neg_deNorm(wh_pred_2024, wh_min, wh_max);
wh_preds = array2table(wh_preds);
wh_preds = [wh_labels wh_preds];
wh_preds.Properties.VariableNames = wh_pred_2024_header;
writetable(wh_preds,'wh_multiyear_test.txt','Delimiter',' ')  
