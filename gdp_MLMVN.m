clear;

%read in dataset
G = readtable('gdp.csv');

%cleanup
g = G;
g(:,1) = []; %index read in as a col
g_header = g(1,:);
g(1,:) = []; %remove header row for processing
g_labels = g(:,1);
g(:,1) = [];%remove labels for processing

%normalize the dataset to the unit circle
g = table2array(g);
g = knnimpute(g);
g_min = min(g(:));
g_max = max(g(:));
g_res = g;
%g_norm = ( (g - min(g(:)) ) / ( max(g(:)) - min(g(:)) ) ) .* 3*pi/2 + pi/4;
g = neg_norm(g);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(g,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = g(trainId,:);
testData = g(testId,:);
num_inputs = size(trainData);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

actuals_2023 = actual_output;
orig_2023 = desired_output;
filename = "2023_Validity_GDP";
var_to_txt(orig_2023, actuals_2023, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(g, hidneur_weights1, hidneur_weights2, outneur_w);

%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};


g_pred_2023_input = array2table(g);
g_pred_2023 = [g_labels g_pred_2023_input];
%u_pred_2023_input.Properties.VariableNames = u_header;
g_header1 = table2cell(g_header);
g_header1 = string(g_header1);
allVars1 = 1:width(g_pred_2023);
g_pred_2023 = renamevars(g_pred_2023,allVars1, g_header1);
g_pred_2023_header = g_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
g_pred_2023(:,1) = []; %remove labels
g_pred_2023 = table2array(g_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(g_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = g_pred_2023(trainId,:);
testData = g_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.01);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


actuals_2024 = actual_output;
orig_2024 = desired_output;
filename = "2024_Validity_GDP";
var_to_txt(orig_2024, actuals_2024, filename);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(g_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

%col of preds
pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};

%add back to table
g_pred_2023_input = array2table(g_pred_2023);
g_pred_2023_input = [g_labels g_pred_2023_input];
g_pred_2023_input.Properties.VariableNames = g_pred_2023_header;
g_pred_2024 = [g_pred_2023_input pred_val];
g_pred_2024_header = g_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g_pred_2024(:,1) = [];
g_pred_2024 = table2array(g_pred_2024);
g_preds = neg_deNorm(g_pred_2024,g_min,g_max);
g_preds = array2table(g_preds);
g_preds = [g_labels g_preds];
g_preds.Properties.VariableNames = g_pred_2024_header;
writetable(g_preds,'gdp_predictions_updated1.txt','Delimiter',' ')  

