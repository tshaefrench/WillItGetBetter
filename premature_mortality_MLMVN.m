clear;

%read in dataset
P = readtable('premature_mortality_clean.csv');

%remove columns with text
p = P;
p_header = p(1,:); %reserve header for rejoin
p(1,:) = [];%rows
p_labels = p(:,1); % reserve for join later minus header row
p(:,1) = []; %cols

%normalize the dataset for MLMVN
p = table2array(p);
p_res = p;
p = ( (p - min(p(:)) ) / ( max(p(:)) - min(p(:)) ) ) .* 3*pi/2 + pi/4;

%impute nulls
p = knnimpute(p);

%train/test split
% Partiion with 25% data as testing 
hpartition = cvpartition(size(p,1),'Holdout',0.25); 
% Extract indices for training and test 
trainId = training(hpartition);
testId = test(hpartition);
% Use Indices to parition the matrix  
trainData = p(trainId,:);
testData = p(testId,:);

%%%%%%Train and Test%%%%%%%%%%
%Training the Network
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [32 64],0.000335);
pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
% ang_RMSE - resulting angular RMSE

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(p, hidneur_weights1, hidneur_weights2, outneur_w);

p = deNorm(p, p_res);

%rejoin datasets
%make vars align
allVars = 1:width(p_header);
newNames = append("Var",string(allVars));
p_header = renamevars(p_header,allVars,newNames);

%rejoin row labels and dataset
p = array2table(p);
p_ts_rejoin = [p_labels p];

%make vars align for header labels
allVars1 = 1:width(p_ts_rejoin);
newNames1 = append("Var",string(allVars1));
p_ts_rejoin = renamevars(p_ts_rejoin,allVars1,newNames1);
p_ts_final = [p_header;p_ts_rejoin];

%denormalize the predicted values
pred_val = deNorm(pred_val, p_res);
pred_val = num2cell(pred_val);
pred_col = [{'predictions'};pred_val]; 

%rejoin dataset and predicted column
p_ts_final = [p_ts_final pred_col];
writetable(p_ts_final,'prem_mort_predictions.txt','Delimiter',' ')  

