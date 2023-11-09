clear;

%read in dataset as cell for later processing
W = readtable('women_parliament.csv');

%drop first column with text
w = W;
w(:,1) = []; %index
w_header = w(1,:); 
w(1,:) = []; 
w_labels = w(:,1);
w(:,1) = [];


%normalize the dataset for MLMVN
w = table2array(w);
w_res = w;
w = ( (w - min(w(:)) ) / ( max(w(:)) - min(w(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
w = knnimpute(w);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(w,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = w(trainId,:);
testData = w(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.009755125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(w, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};

%add header back and join pred column, reserve header row.
w_input = array2table(w);
w_pred_2022 = [w_labels w_input];
w_header1 = table2cell(w_header);
w_header1 = string(w_header1);
allVars1 = 1:width(w_pred_2022);
w_pred_2022 = renamevars(w_pred_2022,allVars1, w_header1);
w_pred_2022 = [w_pred_2022 pred_val];
w_pred_2022_header = w_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
w_pred_2022(:,1) = []; %remove labels
w_pred_2022 = table2array(w_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(w_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = w_pred_2022(trainId,:);
testData = w_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[128 256],0.012426125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(w_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
w_pred_2022_input = array2table(w_pred_2022);
w_pred_2022_input = [w_labels w_pred_2022_input];
w_pred_2022_input.Properties.VariableNames = w_pred_2022_header;
w_pred_2023 = [w_pred_2022_input pred_val];
w_pred_2023_header = w_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
w_pred_2023(:,1) = []; %remove labels
w_pred_2023 = table2array(w_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(w_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = w_pred_2023(trainId,:);
testData = w_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[128 256],0.012426125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(w_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
w_pred_2023_input = array2table(w_pred_2023);
w_pred_2023_input = [w_labels w_pred_2023_input];
w_pred_2023_input.Properties.VariableNames = w_pred_2023_header;
w_pred_2024 = [w_pred_2023_input pred_val];
w_pred_2024_header = w_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w_pred_2024(:,1) = [];
w_pred_2024 = table2array(w_pred_2024);
w_preds = new_deNorm(w_pred_2024, w_res);
w_preds = array2table(w_preds);
w_preds = [w_labels w_preds];
w_preds.Properties.VariableNames = w_pred_2024_header;
writetable(w_preds,'women_parliament_predictions_updated.txt','Delimiter',' ')  

