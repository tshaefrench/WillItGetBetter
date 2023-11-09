clear;

%read in disposable income dataset
A = readtable('avg_wage.csv');

%reserve the text data for joining later. drop header col & row
a =A;
a_header = a(1,:); %reserve header for join later
a(1,:) = []; %rows
a_labels = a(:,1); %reserve for join later minus header row
a(:,1) = [];%cols

%normalize the dataset for neural network
a = table2array(a);
a_res = a; %reserve for denorm later
a = ( (a - min(a(:)) ) / ( max(a(:)) - min(a(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls with nearest neighbor method
a = knnimpute(a);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(a,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = a(trainId,:);
testData = a(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32], 0.00325325);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(a, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};


%add header back and join pred column, reserve header row.
a_input = array2table(a);
a_pred_2022 = [a_labels a_input];
a_header1 = table2cell(a_header);
a_header1 = string(a_header1);
allVars1 = 1:width(a_pred_2022);
a_pred_2022 = renamevars(a_pred_2022,allVars1, a_header1);
a_pred_2022 = [a_pred_2022 pred_val];
a_pred_2022_header = a_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
a_pred_2022(:,1) = []; %remove labels
a_pred_2022 = table2array(a_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(a_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = a_pred_2022(trainId,:);
testData = a_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32], 0.00325325);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(a_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
a_pred_2022_input = array2table(a_pred_2022);
a_pred_2022_input = [a_labels a_pred_2022_input];
a_pred_2022_input.Properties.VariableNames = a_pred_2022_header;
a_pred_2023 = [a_pred_2022_input pred_val];
a_pred_2023_header = a_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
a_pred_2023(:,1) = []; %remove labels
a_pred_2023 = table2array(a_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(a_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = a_pred_2023(trainId,:);
testData = a_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32], 0.00325325);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(a_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
a_pred_2023_input = array2table(a_pred_2023);
a_pred_2023_input = [a_labels a_pred_2023_input];
a_pred_2023_input.Properties.VariableNames = a_pred_2023_header;
a_pred_2024 = [a_pred_2023_input pred_val];
a_pred_2024_header = a_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_pred_2024(:,1) = [];
a_pred_2024 = table2array(a_pred_2024);
a_preds = deNorm(a_pred_2024, a_res);
a_preds = array2table(a_preds);
a_preds = [a_labels a_preds];
a_preds.Properties.VariableNames = a_pred_2024_header;
writetable(a_preds,'avg_wage_multiyear_predictions.txt','Delimiter',' ')  
