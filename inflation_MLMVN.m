clear;

%read in inflation dataset from worldbank.org delete null lines of data.
N = readtable('inflation_cleaner.csv');

%reserve the text data for joining later. drop header col & row
n = N;
n_header = n(1,:); %reserve header for join later
n(1,:) = []; %remove header row
n_labels = n(:,1:2); %reserve for join later minus header row
%drop text values & pred
%n(:, 34) = [];
n(:,1:2) = [];

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
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData,[64 128],0.012426125);

pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(trainData, hidneur_weights1, hidneur_weights2, outneur_w);

% Predictions on entire dataset from trained network
[pred_val] = Net_pred(n, hidneur_weights1, hidneur_weights2, outneur_w);

n = deNorm(n, n_res);

%rejoin datasets
%make vars align
allVars = 1:width(n_header);
newNames = append("Var",string(allVars));
n_header = renamevars(n_header,allVars,newNames);

%rejoin row labels and dataset
n = array2table(n);
n_ts_rejoin = [n_labels n];

%make vars align for header labels
allVars1 = 1:width(n_ts_rejoin);
newNames1 = append("Var",string(allVars1));
n_ts_rejoin = renamevars(n_ts_rejoin,allVars1,newNames1);
n_ts_final = [n_header;n_ts_rejoin];

%denormalize the predicted values
pred_val = deNorm(pred_val, n_res);
pred_val = num2cell(pred_val);
pred_col = [{'predictions'};pred_val]; 

%rejoin dataset and predicted column
n_ts_final = [n_ts_final pred_col];
writetable(n_ts_final,'inflation_predictions.txt','Delimiter',' ')  

