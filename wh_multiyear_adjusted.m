clear;

%read in dataset
WH = readtable('world_happiness_adjusted.csv');

%remove columns with text
wh = WH;
wh(:,1) = []; %incorrect col
wh_header = wh.Properties.VariableNames; %get header row from vars. 
wh_labels = wh(:,1);
wh(:,1) = []; %cols

%normalize the dataset for MLMVN
wh = table2array(wh);
wh_res = wh;
wh = ( (wh - min(wh(:)) ) / ( max(wh(:)) - min(wh(:)) ) ) .* 3*pi/2 + pi/4;
wh = (wh - min(wh(:))) / (max(wh(:)) - min(wh(:)));

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

%%%%%%%%UPDATE THIS!!!!!%%%%%%%%%%%%
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};

%add header back and join predicted column and table
wh = array2table(wh);
wh = [wh_labels wh];
wh.Properties.VariableNames = wh_header;
wh_pred_2024 = [wh pred_val];
%wh_pred_2024_res = wh_pred_2024;

% %make vars align for header labels
wh_header_2024 = wh_pred_2024.Properties.VariableNames;

%%%%%%%%%%END OF UPDATE%%%%%%%%%%%%
wh_pred_2024(:,1) = []; %cols

wh_pred_2024 = table2array(wh_pred_2024);

%train/test split
% Partition with 30% data as testing 
hpartition = cvpartition(size(wh_pred_2024,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = wh_pred_2024(trainId,:);
testData = wh_pred_2024(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network for 2nd run
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.0014838125);
pause(3)

% Testing of the trained network for 2nd run
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(wh_pred_2024, hidneur_weights1, hidneur_weights2, outneur_w);

%denormalize the table and predicted column
%wh_my_pred = new_deNorm(wh_pred_2024, wh_res);
%pred_val = new_deNorm(pred_val, wh_res);

wh_pred_2024 = (wh_pred_2024 - pi/4) / (3*pi/2);
wh_pred_2024 = wh_pred_2024 * (max(wh_res(:)) - min(wh_res(:))) + min(wh_res(:));

pred_val = (pred_val - pi/4) / (3*pi/2);
pred_val = pred_val * (max(wh_res(:)) - min(wh_res(:))) + min(wh_res(:));

%rejoin row labels and dataset
wh_my_pred = array2table(wh_my_pred); %needs to be table to adjust variables.
wh_2024_withpreds = [wh_labels wh_my_pred]; %add row labels back to accommodate header variables
wh_2024_withpreds.Properties.VariableNames = wh_header_2024;

%take predicted values. adjust column label
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Prediction'};

%rejoin dataset and predicted column
wh_ts_final_24 = [wh_2024_withpreds pred_val];
writetable(wh_ts_final_24,'world_happiness_predictions_multiyear_updated.txt','Delimiter',' ')  