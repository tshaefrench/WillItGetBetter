clear;

%read in average wage dataset
A = readtable('avg_wage.csv');

%reserve the text data for joining later. drop header col & row
a =A;
a_labels = a(:,1); %reserve row labels for join later
a(:,1) = [];%1st col
a_header = a(1,:);
a(1,:) = [];%1st row

%normalize the dataset for neural network
%normalize to 3/4 of the unit circle
a = table2array(a);
a_res = a; %reserve for denorm later
a = ( (a - min(a(:)) ) / ( max(a(:)) - min(a(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls with nearest neighbor method
a = knnimpute(a);

%train/test split
% Partition with 25% data as testing 
hpartition = cvpartition(size(a,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = a(trainId,:);
testData = a(testId,:);

%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [16 32], 0.00325325);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(a, hidneur_weights1, hidneur_weights2, outneur_w);

a = deNorm(a, a_res);

%denormalize the predicted values
pred_val = deNorm(pred_val, a_res);
pred_val = num2cell(pred_val);
pred_col = [{'predictions'};pred_val]; 

%rejoin datasets
%make vars align
allVars = 1:width(a_header);
newNames = append("Var",string(allVars));
a_header = renamevars(a_header,allVars,newNames);

%rejoin row labels and dataset
a = array2table(a);
a_ts_rejoin = [a_labels a];

%make vars align for header labels
allVars1 = 1:width(a_ts_rejoin);
newNames1 = append("Var",string(allVars1));
a_ts_rejoin = renamevars(a_ts_rejoin,allVars1,newNames1);
a_ts_final = [a_header;a_ts_rejoin];

%denormalize the predicted values
%pred_val = deNorm(pred_val, n_res);
pred_val = num2cell(pred_val);
pred_col = [{'predictions'};pred_val]; 

%rejoin dataset and predicted column
a_ts_final = [a_ts_final pred_col];
writetable(a_ts_final,'avg_wage_predictions.txt','Delimiter',' ')  