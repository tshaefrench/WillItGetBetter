clear;

%read in dataset
WH = readtable('world_happiness_longitudinal.csv');

%remove columns with text
wh = WH;
wh(:,1) = []; %incorrect col
wh_header = wh.Properties.VariableNames; %get header row from vars. 
wh_labels = wh(:,1);
wh(:,1) = []; %cols

%normalize the dataset for MLMVN
wh = table2array(wh);
wh_res = wh;
wh = ( (wh - min(wh(:)) ) / ( max(wh(:)) - min(wh(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
wh = knnimpute(wh);

%train/test split
% Partiion with 30% data as testing 
hpartition = cvpartition(size(wh,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = wh(trainId,:);
testData = wh(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [128 256],0.0014838125);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(wh, hidneur_weights1, hidneur_weights2, outneur_w);

wh = deNorm(wh, wh_res);

%denormalize the predicted values
pred_val = deNorm(pred_val, wh_res);
% pred_val = num2cell(pred_val);
% pred_col = [{'predictions'};pred_val]; 
% pred_col = cell2table(pred_col);

%rejoin datasets
%make vars align
% allVars = 1:width(wh_header);
% newNames = append("Var",string(allVars));
% wh_header = renamevars(wh_header,allVars,newNames);

%rejoin row labels and dataset
wh = array2table(wh);
wh_ts_rejoin = [wh_labels wh];

pred_val = array2table(pred_val);
pred_val.Properties.VariableNames = {'predictions'};

% %make vars align for header labels
wh_ts_rejoin.Properties.VariableNames = wh_header;

%rejoin dataset and predicted column
wh_ts_final = [wh_ts_rejoin pred_val];
writetable(wh_ts_final,'world_happiness_predictions.txt','Delimiter',' ')  
