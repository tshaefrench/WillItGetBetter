clear;

%read in dataset
P = readtable('premature_mortality_long.csv');

%remove columns with text
p = P;
p(2,:) =[];
p_header =  p(1,:);%header first
p(1,:) = [;];
p_labels = p(:,1); % reserve for join later minus header row
p(:,1) = []; %cols



%normalize the dataset for MLMVN
p = table2array(p);
p_res = p;
p = ( (p - min(p(:)) ) / ( max(p(:)) - min(p(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
p = knnimpute(p);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(p,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p(trainId,:);
testData = p(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [32 64],0.0035);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(p, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2021_Predicted'};

%add header back and join pred column, reserve header row.
p_input = array2table(p);
p_pred_2021 = [p_labels p_input];
p_header1 = table2cell(p_header);
p_header1 = string(p_header1);
allVars1 = 1:width(p_pred_2021);
p_pred_2021 = renamevars(p_pred_2021,allVars1, p_header1);
%p_pred_2021.Properties.VariableNames = p_header;
p_pred_2021 = [p_pred_2021 pred_val];
p_pred_2021_header = p_pred_2021.Properties.VariableNames;

%%%%%%%%%%%2022%%%%%%%%%%%%
p_pred_2021(:,1) = []; %remove labels
p_pred_2021 = table2array(p_pred_2021);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(p_pred_2021,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p_pred_2021(trainId,:);
testData = p_pred_2021(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] =  Net_learn(trainData, [32 64],0.0035);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(p_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};

%add header back and join pred column, reserve header row.
p_input = array2table(p_pred_2021);
p_pred_2022 = [p_labels p_input];
%p_pred_2021_header1 = table2cell(p_header);
% p_pred_2021_header1 = string(p_pred_2021_header);
% allVars1 = 1:width(p_pred_2022);
% p_pred_2022 = renamevars(p_pred_2022,allVars1, p_pred_2021_header1);
p_pred_2022.Properties.VariableNames = p_pred_2021_header;
p_pred_2022 = [p_pred_2022 pred_val];
p_pred_2022_header = p_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
p_pred_2022(:,1) = []; %remove labels
p_pred_2022 = table2array(p_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(p_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p_pred_2022(trainId,:);
testData = p_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [32 64],0.0035);
% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(p_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
p_pred_2022_input = array2table(p_pred_2022);
p_pred_2022_input = [p_labels p_pred_2022_input];
p_pred_2022_input.Properties.VariableNames = p_pred_2022_header;
p_pred_2023 = [p_pred_2022_input pred_val];
p_pred_2023_header = p_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
p_pred_2023(:,1) = []; %remove labels
p_pred_2023 = table2array(p_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(p_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p_pred_2023(trainId,:);
testData = p_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] =  Net_learn(trainData, [32 64],0.0035);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(p_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
p_pred_2023_input = array2table(p_pred_2023);
p_pred_2023_input = [p_labels p_pred_2023_input];
p_pred_2023_input.Properties.VariableNames = p_pred_2023_header;
p_pred_2024 = [p_pred_2023_input pred_val];
p_pred_2024_header = p_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_pred_2024(:,1) = [];
p_pred_2024 = table2array(p_pred_2024);
p_preds = new_deNorm(p_pred_2024, p_res);
p_preds = array2table(p_preds);
p_preds = [p_labels p_preds];
p_preds.Properties.VariableNames = p_pred_2024_header;
writetable(p_preds,'prem_mortality_multiyear_predictions_updated.txt','Delimiter',' ')  
