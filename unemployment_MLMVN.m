ufclear;

%read in dataset
U = readtable('unemployment.csv');

%remove columns with text
U{1,1} = cellstr('Year');
u = U;
%u(1,:) = []; %row filled with index numbers
u_header = u(1,:);
u(1,:) = []; %rows
u_labels = u(:,1);
u(:,1) = []; %cols

%normalize the dataset for MLMVN
u = table2array(u);
u_res = u;
u_min = min(u(:));
u = ( (u - min(u(:)) ) / ( max(u(:)) - min(u(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
u = knnimpute(u);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(u,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = u(trainId,:);
testData = u(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.0036518);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(u, hidneur_weights1, hidneur_weights2, outneur_w);
%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};


u_pred_2023_input = array2table(u);
u_pred_2023 = [u_labels u_pred_2023_input];
%u_pred_2023_input.Properties.VariableNames = u_header;
u_header1 = table2cell(u_header);
u_header1 = string(u_header1);
allVars1 = 1:width(u_pred_2023);
u_pred_2023 = renamevars(u_pred_2023,allVars1, u_header1);
u_pred_2023 = [u_pred_2023 pred_val];
u_pred_2023_header = u_pred_2023.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
u_pred_2023(:,1) = []; %remove labels
u_pred_2023 = table2array(u_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(u_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = u_pred_2023(trainId,:);
testData = u_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.0036518);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(u_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);
%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};

%add back to table
u_pred_2023_input = array2table(u_pred_2023);
u_pred_2023_input = [u_labels u_pred_2023_input];
u_pred_2023_input.Properties.VariableNames = u_pred_2023_header;
u_pred_2024 = [u_pred_2023_input pred_val];
u_pred_2024_header = u_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u_pred_2024(:,1) = [];
u_pred_2024 = table2array(u_pred_2024);
u_preds = new_deNorm(u_pred_2024, u_res);
u_preds = array2table(u_preds);
u_preds = [u_labels u_preds];
u_preds.Properties.VariableNames = u_pred_2024_header;
writetable(u_preds,'unemployment_predictions_updated.txt','Delimiter',' ')  
