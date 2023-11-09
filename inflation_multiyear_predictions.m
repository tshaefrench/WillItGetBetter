clear;

%read in inflation dataset from worldbank.org delete null lines of data.
N = readtable('inflation_cleaner.csv');

%reserve the text data for joining later. drop header col & row
n = N;
n(:,1) = []; %remove index column
n_header = n(1,:); %reserve header for join later
n(1,:) = []; %remove header row
n_labels = n(:,1); %reserve for join later minus header row
%drop text values & pred
%n(:, 34) = [];
n(:,1) = [];

%normalize the dataset for MLMVN
n = table2array(n);
n_res = n;
n = ( (n - min(n(:)) ) / ( max(n(:)) - min(n(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
n = knnimpute(n);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(n,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = n(trainId,:);
testData = n(testId,:);

%%%%%%Train, Test, Predict%%%%%%%%%%%

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[128 256],0.034426125);

pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(trainData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(n, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};

%add header back and join pred column, reserve header row.
n_input = array2table(n);
n_pred_2022 = [n_labels n_input];
n_header1 = table2cell(n_header);
n_header1 = string(n_header1);
allVars1 = 1:width(n_pred_2022);
n_pred_2022 = renamevars(n_pred_2022,allVars1, n_header1);
n_pred_2022 = [n_pred_2022 pred_val];
n_pred_2022_header = n_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
n_pred_2022(:,1) = []; %remove labels
n_pred_2022 = table2array(n_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(n_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = n_pred_2022(trainId,:);
testData = n_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[128 256],0.034426125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(n_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
n_pred_2022_input = array2table(n_pred_2022);
n_pred_2022_input = [n_labels n_pred_2022_input];
n_pred_2022_input.Properties.VariableNames = n_pred_2022_header;
n_pred_2023 = [n_pred_2022_input pred_val];
n_pred_2023_header = n_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
n_pred_2023(:,1) = []; %remove labels
n_pred_2023 = table2array(n_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(n_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = n_pred_2023(trainId,:);
testData = n_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[128 256],0.034426125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(n_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
n_pred_2023_input = array2table(n_pred_2023);
n_pred_2023_input = [n_labels n_pred_2023_input];
n_pred_2023_input.Properties.VariableNames = n_pred_2023_header;
n_pred_2024 = [n_pred_2023_input pred_val];
n_pred_2024_header = n_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_pred_2024(:,1) = [];
n_pred_2024 = table2array(n_pred_2024);
n_preds = deNorm(n_pred_2024, n_res);
n_preds = array2table(n_preds);
n_preds = [n_labels n_preds];
n_preds.Properties.VariableNames = n_pred_2024_header;
writetable(n_preds,'inflation_multiyear_predictions.txt','Delimiter',' ')  
