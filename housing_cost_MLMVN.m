clear;
%read in hate crimes data
Hg = readtable('housing_cost.csv');

%remove text fields
hg = Hg;
hg(:,2:11) = []; %drop nan cols
hg_header = hg(1,:);
hg(1,:) = [];
hg_labels = hg(:,1);
hg(:,1) = [];

%normalize the dataset for MLMVN
hg = table2array(hg);
hg_res = hg; %for min/max later
hg = ( (hg - min(hg(:)) ) / ( max(hg(:)) - min(hg(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
hg = knnimpute(hg);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(hg,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = hg(trainId,:);
testData = hg(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [256 512],0.00845375);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(hg, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};

%add header back and join pred column, reserve header row.
hg_input = array2table(hg);
hg_pred_2022 = [hg_labels hg_input];
hg_header1 = table2cell(hg_header);
hg_header1 = string(hg_header1);
allVars1 = 1:width(hg_pred_2022);
hg_pred_2022 = renamevars(hg_pred_2022,allVars1, hg_header1);
hg_pred_2022 = [hg_pred_2022 pred_val];
hg_pred_2022_header = hg_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
hg_pred_2022(:,1) = []; %remove labels
hg_pred_2022 = table2array(hg_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(hg_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = hg_pred_2022(trainId,:);
testData = hg_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [256 512],0.00845375);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(hg_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
hg_pred_2022_input = array2table(hg_pred_2022);
hg_pred_2022_input = [hg_labels hg_pred_2022_input];
hg_pred_2022_input.Properties.VariableNames = hg_pred_2022_header;
hg_pred_2023 = [hg_pred_2022_input pred_val];
hg_pred_2023_header = hg_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
hg_pred_2023(:,1) = []; %remove labels
hg_pred_2023 = table2array(hg_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(hg_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = hg_pred_2023(trainId,:);
testData = hg_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [256 512],0.00845375);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(hg_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
hg_pred_2023_input = array2table(hg_pred_2023);
hg_pred_2023_input = [hg_labels hg_pred_2023_input];
hg_pred_2023_input.Properties.VariableNames = hg_pred_2023_header;
hg_pred_2024 = [hg_pred_2023_input pred_val];
hg_pred_2024_header = hg_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hg_pred_2024(:,1) = [];
hg_pred_2024 = table2array(hg_pred_2024);
hg_preds = new_deNorm(hg_pred_2024, hg_res);
hg_preds = array2table(hg_preds);
hg_preds = [hg_labels hg_preds];
hg_preds.Properties.VariableNames = hg_pred_2024_header;
writetable(hg_preds,'housing_cost_multiyear_predictions_updated.txt','Delimiter',' ')  
