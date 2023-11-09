clear;

Gn = readtable('gender_equality_longitudinal.csv');

%remove columns with text
gn = Gn;
gn(:,1) = []; %row filled with index numbers
gn_header = gn(1,:);
gn(1,:) = []; %rows
gn_labels = gn(:,1:2);
gn(:,1:2) = []; %cols

%normalize data
%normalize the dataset for neural network
gn = table2array(gn);
gn_res = gn;
gn = ( (gn - min(gn(:)) ) / ( max(gn(:)) - min(gn(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
gn = knnimpute(gn);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(gn,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn(trainId,:);
testData = gn(testId,:);

%%%%%%Train and Test%%%%%%%%%%

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn, hidneur_weights1, hidneur_weights2, outneur_w);

% 
% %normalize data
% %normalize the dataset for neural network
% gn = table2array(gn);
% gn_res = gn;
% gn = ( (gn - min(gn(:)) ) / ( max(gn(:)) - min(gn(:)) ) ) .* 3*pi/2 + pi/4;
% 
% %impute nulls
% gn = knnimpute(gn);
% 
% %train/test split
% % Partiion with 25% data as testing 
% hpartition = cvpartition(size(gn,1),'Holdout',0.25); 
% % Extract indices for training and test 
% trainId = training(hpartition);
% testId = test(hpartition);
% % Use Indices to parition the matrix  
% trainData = gn(trainId,:);
% testData = gn(testId,:);
% 
% %%%%%%Train and Test%%%%%%%%%%
% 
% %Training the Network
% [hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
% pause(3)
% 
% % Testing of the trained network
% [ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% % ang_RMSE - resulting angular RMSE
% 
% % Predictions on entire dataset from trained network
% [pred_val] = Net_pred(gn, hidneur_weights1, hidneur_weights2, outneur_w);


pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2014_Predicted'};

%add header back and join pred column, reserve header row.
gn_input = array2table(gn);
gn_pred_2014 = [gn_labels gn_input];
gn_header1 = table2cell(gn_header);
gn_header1 = string(gn_header1);
allVars1 = 1:width(gn_pred_2014);
gn_pred_2014 = renamevars(gn_pred_2014,allVars1, gn_header1);
gn_pred_2014 = [gn_pred_2014 pred_val];
gn_pred_2014_header = gn_pred_2014.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2015 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2014(:,1:2) = []; %remove labels
gn_pred_2014 = table2array(gn_pred_2014);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2014,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2014(trainId,:);
testData = gn_pred_2014(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2014, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2015_Predicted'};
gn_pred_2014_input = array2table(gn_pred_2014);
gn_pred_2014_input = [gn_labels gn_pred_2014_input];
gn_pred_2014_input.Properties.VariableNames = gn_pred_2014_header;
gn_pred_2015 = [gn_pred_2014_input pred_val];
gn_pred_2015_header = gn_pred_2015.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2016 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2015(:,1:2) = []; %remove labels
gn_pred_2015 = table2array(gn_pred_2015);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2015,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2015(trainId,:);
testData = gn_pred_2015(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2015, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2016_Predicted'};
gn_pred_2015_input = array2table(gn_pred_2015);
gn_pred_2015_input = [gn_labels gn_pred_2015_input];
gn_pred_2015_input.Properties.VariableNames = gn_pred_2015_header;
gn_pred_2016 = [gn_pred_2015_input pred_val];
gn_pred_2016_header = gn_pred_2016.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2017 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2016(:,1:2) = []; %remove labels
gn_pred_2016 = table2array(gn_pred_2016);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2016,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2016(trainId,:);
testData = gn_pred_2016(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2016, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2017_Predicted'};
gn_pred_2016_input = array2table(gn_pred_2016);
gn_pred_2016_input = [gn_labels gn_pred_2016_input];
gn_pred_2016_input.Properties.VariableNames = gn_pred_2016_header;
gn_pred_2017 = [gn_pred_2016_input pred_val];
gn_pred_2017_header = gn_pred_2017.Properties.VariableNames;

%%%%%%%%%%%%%%%%2018 Preds%%%%%%%%%%%%%%%%%%%%%%%%%

gn_pred_2017(:,1:2) = []; %remove labels
gn_pred_2017 = table2array(gn_pred_2017);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2017,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2017(trainId,:);
testData = gn_pred_2017(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2017, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2018_Predicted'};

%add header back and join pred column, reserve header row.
gn_pred_2017_input = array2table(gn_pred_2017);
gn_pred_2017_input = [gn_labels gn_pred_2017_input];
gn_pred_2017_input.Properties.VariableNames = gn_pred_2017_header;
gn_pred_2018 = [gn_pred_2017_input pred_val];
gn_pred_2018_header = gn_pred_2018.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2019 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2018(:,1:2) = []; %remove labels
gn_pred_2018 = table2array(gn_pred_2018);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2018,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2018(trainId,:);
testData = gn_pred_2018(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);


% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2018, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2019_Predicted'};
gn_pred_2018_input = array2table(gn_pred_2018);
gn_pred_2018_input = [gn_labels gn_pred_2018_input];
gn_pred_2018_input.Properties.VariableNames = gn_pred_2018_header;
gn_pred_2019 = [gn_pred_2018_input pred_val];
gn_pred_2019_header = gn_pred_2019.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2020 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2019(:,1:2) = []; %remove labels
gn_pred_2019 = table2array(gn_pred_2019);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2019,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2019(trainId,:);
testData = gn_pred_2019(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2019, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2020_Predicted'};
gn_pred_2019_input = array2table(gn_pred_2019);
gn_pred_2019_input = [gn_labels gn_pred_2019_input];
gn_pred_2019_input.Properties.VariableNames = gn_pred_2019_header;
gn_pred_2020 = [gn_pred_2019_input pred_val];
e_pred_2020_header = gn_pred_2020.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2021 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2020(:,1:2) = []; %remove labels
gn_pred_2020 = table2array(gn_pred_2020);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2020,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2020(trainId,:);
testData = gn_pred_2020(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2020, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2021_Predicted'};
gn_pred_2020_input = array2table(gn_pred_2020);
gn_pred_2020_input = [gn_labels gn_pred_2020_input];
gn_pred_2020_input.Properties.VariableNames = e_pred_2020_header;
gn_pred_2021 = [gn_pred_2020_input pred_val];
gn_pred_2021_header = gn_pred_2021.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2022 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2021(:,1:2) = []; %remove labels
gn_pred_2021 = table2array(gn_pred_2021);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2021,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2021(trainId,:);
testData = gn_pred_2021(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2021, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2022_Predicted'};
gn_pred_2021_input = array2table(gn_pred_2021);
gn_pred_2021_input = [gn_labels gn_pred_2021_input];
gn_pred_2021_input.Properties.VariableNames = gn_pred_2021_header;
gn_pred_2022 = [gn_pred_2021_input pred_val];
gn_pred_2022_header = gn_pred_2022.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%2023 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2022(:,1:2) = []; %remove labels
gn_pred_2022 = table2array(gn_pred_2022);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2022,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2022(trainId,:);
testData = gn_pred_2022(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2022, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2023_Predicted'};
gn_pred_2022_input = array2table(gn_pred_2022);
gn_pred_2022_input = [gn_labels gn_pred_2022_input];
gn_pred_2022_input.Properties.VariableNames = gn_pred_2022_header;
gn_pred_2023 = [gn_pred_2022_input pred_val];
gn_pred_2023_header = gn_pred_2023.Properties.VariableNames;


%%%%%%%%%%%%%%%%%%2024 Year Pred%%%%%%%%%%%%%%%%
gn_pred_2023(:,1:2) = []; %remove labels
gn_pred_2023 = table2array(gn_pred_2023);
%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(gn_pred_2023,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = gn_pred_2023(trainId,:);
testData = gn_pred_2023(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.0998);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(gn_pred_2023, hidneur_weights1, hidneur_weights2, outneur_w);

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'2024_Predicted'};
gn_pred_2023_input = array2table(gn_pred_2023);
gn_pred_2023_input = [gn_labels gn_pred_2023_input];
gn_pred_2023_input.Properties.VariableNames = gn_pred_2023_header;
gn_pred_2024 = [gn_pred_2023_input pred_val];
gn_pred_2024_header = gn_pred_2024.Properties.VariableNames;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%final%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gn_pred_2024(:,1:2) = [];
gn_pred_2024 = table2array(gn_pred_2024);
gn_preds = new_deNorm(gn_pred_2024, gn_res);
gn_preds = array2table(gn_preds);
gn_preds = [gn_labels gn_preds];
gn_preds.Properties.VariableNames = gn_pred_2024_header;
writetable(gn_preds,'gender_equality_multiyear_predictions_updated.txt','Delimiter',' ')  

