clear;

%read in education attainment data
E = readtable('education_data.csv');

%remove first column and row because text data
e = E;
e_header = e(1,:); %reserve header for join later
e(1,:) = []; %rows
e_labels = e(:,1); %reserve for join later minus header row
e(:,1) = [];

%normalize data
%normalize the dataset for neural network
e = table2array(e);
e_res = e;
e = ( (e - min(e(:)) ) / ( max(e(:)) - min(e(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
e = knnimpute(e);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(e,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = e(trainId,:);
testData = e(testId,:);

%%%%%%Train and Test%%%%%%%%%%

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [64 128],0.018289);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(e, hidneur_weights1, hidneur_weights2, outneur_w);


%%%%%%%%%inprogress%%%%%%%%%%%%%%%


e = deNorm(e, e_res);

%rejoin datasets
%make vars align
allVars = 1:width(e_header);
newNames = append("Var",string(allVars));
e_header = renamevars(e_header,allVars,newNames);

%rejoin row labels and dataset
e = array2table(e);
e_ts_rejoin = [e_labels e];

%make vars align for header labels
allVars1 = 1:width(e_ts_rejoin);
newNames1 = append("Var",string(allVars1));
e_ts_rejoin = renamevars(e_ts_rejoin,allVars1,newNames1);
e_ts_final = [e_header;e_ts_rejoin];

%denormalize the predicted values
pred_val = deNorm(pred_val, e_res);
pred_val = num2cell(pred_val);
pred_col = [{'predictions'};pred_val]; 

%rejoin dataset and predicted column
e_ts_final = [e_ts_final pred_col];
writetable(e_ts_final,'education_predictions.txt','Delimiter',' ')  
