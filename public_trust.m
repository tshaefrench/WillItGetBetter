clear; 

Pub = readtable('public_trust.csv');

pub = Pub;
pub(1,:) =[]; %drop row with indices labelled
pub_header = pub(1,:);
pub(1,:) = []; %rows
pub_labels = pub(:,1);
pub(:,1) = []; %cols

%normalize the dataset for MLMVN
pub = table2array(pub);
pub_res = pub;
pub = ( (pub - min(pub(:)) ) / ( max(pub(:)) - min(pub(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
pub = knnimpute(pub);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(pub,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = pub(trainId,:);
testData = pub(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.0036518);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(pub, hidneur_weights1, hidneur_weights2, outneur_w);

%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};


pub_pred_2023_input = array2table(pub);
pub_pred_2023 = [pub_labels pub_pred_2023_input];
pub_pred_2023_input.Properties.VariableNames = pub_pred_2023_header;
pub_pred_2023_header = pub_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
pub_pred_2023(:,1) = []; %remove labels
pub_pred_2023 = table2array(pub_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(pub_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = pub_pred_2023(trainId,:);
testData = pub_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [32 64],0.120486);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(pub_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);
%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};

%add back to table
pub_pred_2023_input = array2table(pub_pred_2023);
pub_pred_2023_input = [pub_labels pub_pred_2023_input];
pub_pred_2023_input.Properties.VariableNames = pub_pred_2023_header;
pub_pred_2024 = [pub_pred_2023_input pred_val];
pub_pred_2024_header = pub_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pub_pred_2024(:,1) = [];
pub_pred_2024 = table2array(pub_pred_2024);
pub_preds = deNorm(pub_pred_2024, pub_res);
pub_preds = array2table(pub_preds);
pub_preds = [pub_labels pub_preds];
pub_preds.Properties.VariableNames = pub_pred_2024_header;
writetable(pub_preds,'public_trust_predictions.txt','Delimiter',' ')  
