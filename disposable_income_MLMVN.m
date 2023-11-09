clear;

%read in disposable income dataset
D = readtable('disposable_income.csv');

%reserve the text data for joining later. drop header col & row
d =D;
d_header = d(1,:); %reserve header for join later
d(1,:) = []; %rows
d_labels = d(:,1); %reserve for join later minus header row
d(:,1) = [];%cols

%normalize the dataset for neural network
d = table2array(d);
d_res = d; %reserve for denorm later
d = ( (d - min(d(:)) ) / ( max(d(:)) - min(d(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls with nearest neighbor method
d = knnimpute(d);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(d,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = d(trainId,:);
testData = d(testId,:);
numinput = size(trainData)
%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.01889175);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(d, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};


%add header back and join pred column, reserve header row.
d_input = array2table(d);
d_pred_2022 = [d_labels d_input];
d_header1 = table2cell(d_header);
d_header1 = string(d_header1);
allVars1 = 1:width(d_pred_2022);
d_pred_2022 = renamevars(d_pred_2022,allVars1, d_header1);
d_pred_2022 = [d_pred_2022 pred_val];
d_pred_2022_header = d_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
d_pred_2022(:,1) = []; %remove labels
d_pred_2022 = table2array(d_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(d_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = d_pred_2022(trainId,:);
testData = d_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.01889175);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(d_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
d_pred_2022_input = array2table(d_pred_2022);
d_pred_2022_input = [d_labels d_pred_2022_input];
d_pred_2022_input.Properties.VariableNames = d_pred_2022_header;
d_pred_2023 = [d_pred_2022_input pred_val];
d_pred_2023_header = d_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
d_pred_2023(:,1) = []; %remove labels
d_pred_2023 = table2array(d_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(d_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = d_pred_2023(trainId,:);
testData = d_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32],0.01889175);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(d_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
d_pred_2023_input = array2table(d_pred_2023);
d_pred_2023_input = [d_labels d_pred_2023_input];
d_pred_2023_input.Properties.VariableNames = d_pred_2023_header;
d_pred_2024 = [d_pred_2023_input pred_val];
d_pred_2024_header = d_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_pred_2024(:,1) = [];
d_pred_2024 = table2array(d_pred_2024);
d_preds = new_deNorm(d_pred_2024, d_res);
d_preds = array2table(d_preds);
d_preds = [d_labels d_preds];
d_preds.Properties.VariableNames = d_pred_2024_header;
%writetable(d_preds,'disp_income_multiyear_predictions_updated1.txt','Delimiter',' ')  
