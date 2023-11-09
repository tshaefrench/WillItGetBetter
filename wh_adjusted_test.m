clear;

%read in dataset
WH = readtable('premature_mortality_long.csv');

%remove columns with text
wh = WH;
wh(2,:) =[];
wh_header =  wh(1,:);%header first
wh(1,:) = [;];
wh_labels = wh(:,1); % reserve for join later minus header row
wh(:,1) = []; %cols

%normalize the dataset for MLMVN
wh = table2array(wh);
wh_min = min(wh(:));
wh_max = max(wh(:));
%wh = ( (wh - min(wh(:)) ) / ( max(wh(:)) - min(wh(:)) ) ) .* 3*pi/2 + pi/4;
wh = neg_norm(wh);

%impute nulls
wh = knnimpute(wh);

% %train/test split
% % Partiion with 25% data as testing 
% hpartition = cvpartition(size(wh,1),'Holdout',0.25); 
% % Extract indices for training and test 
% trainId = training(hpartition);
% testId = test(hpartition);
% % Use Indices to parition the matrix  
% trainData = wh(trainId,:);
% testData = wh(testId,:);
% 
% %%%%%%Train and Test%%%%%%%%%%
% %Training the Network
% [hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [32 64],0.0035);
% pause(3)
% 
% % Testing of the trained network
% [ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% % ang_RMSE - resulting angular RMSE
% 
% % Predictions on entire dataset from trained network
% [pred_val] = Net_pred(wh, hidneur_weights1, hidneur_weights2, outneur_w);
% 
% pred_val = array2table(pred_val);
% pred_val.Properties.VariableNames = {'2021_Predicted'};
% 
% %add header back and join pred column, reserve header row.
% p_input = array2table(wh);
% p_pred_2021 = [wh_labels p_input];
% p_header1 = table2cell(wh_header);
% p_header1 = string(p_header1);
% allVars1 = 1:width(p_pred_2021);
% p_pred_2021 = renamevars(p_pred_2021,allVars1, p_header1);
% %p_pred_2021.Properties.VariableNames = p_header;
% p_pred_2021 = [p_pred_2021 pred_val];
% p_pred_2021_header = p_pred_2021.Properties.VariableNames;
% 
% %%%%%%%%%%%2022%%%%%%%%%%%%
% p_pred_2021(:,1) = []; %remove labels
% p_pred_2021 = table2array(p_pred_2021);
% %train/test split
% % Partition with 25% data as testing 
% hpartition = cvpartition(size(p_pred_2021,1),'Holdout',0.25); 
% % Extract indices for training and test 
% trainId = training(hpartition);
% testId = test(hpartition);
% % Use Indices to parition the matrix  
% trainData = p_pred_2021(trainId,:);
% testData = p_pred_2021(testId,:);
% 
% %Training the Network
% [hidneur_weights1, hidneur_weights2, outneur_w, iterations] =  Net_learn(trainData, [32 64],0.0035);
% pause(3)
% 
% % Testing of the trained network
% [ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% % ang_RMSE - resulting angular RMSE
% 
% % Predictions on entire dataset from trained network
% [pred_val] = Net_pred(p_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);
% 
% pred_val = array2table(pred_val);
% pred_val.Properties.VariableNames = {'2022_Predicted'};
% 
% %add header back and join pred column, reserve header row.
% p_input = array2table(p_pred_2021);
% p_pred_2022 = [wh_labels p_input];
% %p_pred_2021_header1 = table2cell(p_header);
% % p_pred_2021_header1 = string(p_pred_2021_header);
% % allVars1 = 1:width(p_pred_2022);
% % p_pred_2022 = renamevars(p_pred_2022,allVars1, p_pred_2021_header1);
% p_pred_2022.Properties.VariableNames = p_pred_2021_header;
% p_pred_2022 = [p_pred_2022 pred_val];
% p_pred_2022_header = p_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
% p_pred_2022(:,1) = []; %remove labels
% p_pred_2022 = table2array(p_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(wh,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = wh(trainId,:);
testData = wh(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.0014838125);
% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(wh, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
wh_input = array2table(wh);
wh_input = [wh_labels wh_input];
wh_header1 = table2cell(wh_header);
wh_input.Properties.VariableNames = wh_header1;
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
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] =  Net_learn(trainData, [128 256],0.0014838125);
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
writetable(wh_preds,'wh_new_denorm_test.txt','Delimiter',' ')  
