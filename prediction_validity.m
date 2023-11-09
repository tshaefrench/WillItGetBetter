%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%Validity%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function[mse, rmse, r_squared, mae, target_range] = prediction_validity(filepath,first_layer, second_layer, threshold)

    %read in average wage dataset
    A = readtable(filepath);
    
    %reserve the text data for joining later. drop header col & row
    a =A;
    % a_header = a(1,:);
    % a(1,:) = [];%1st row
    % a_labels = a(:,1); %reserve row labels for join later
    % a(:,1) = [];%1st col
    
    % remove rows with text values
    row_has_text = any(isnan(str2double(a)), 2);
    a(row_has_text, :) = [];
    
    % remove columns with text values
    col_has_text = any(isnan(str2double(a)), 1);
    a(:, col_has_text) = [];
    
    %normalize the dataset for neural network
    a = table2array(a);
    a = ( (a - min(a(:)) ) / ( max(a(:)) - min(a(:)) ) ) .* 3*pi/2 + pi/4;
    
    %impute nulls with nearest neighbor method
    a = knnimpute(a);
    a_res = a;
    
    %reserve last col for validity calculations
    last_idx = size(a,2);
    actual_vals = a(:,last_idx);
    a(:,last_idx) = [];
    
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
    [hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(trainData, [first_layer second_layer], threshold);
    pause(3)
    
    % Testing of the trained network
    [ang_RMSE, actual_output, desired_output] = Net_test(testData, hidneur_weights1, hidneur_weights2, outneur_w);
    
    % Predictions on entire dataset from trained network
    [pred_val] = Net_pred(a, hidneur_weights1, hidneur_weights2, outneur_w);
    
    a_preds = pred_val;
    a_preds = deNorm(a_preds, a_res);

    filename = "Validity_Vars_For_Testing";
    var_to_txt(actual_vals, a_preds, filename);
    
    % calc mean squared error
    mse = mean((actual_vals - a_preds).^2);

    rmse = sqrt(mse);
    
    % calc variance of actual values
    var_actual = var(actual_vals);

    target_range = range(actual_vals);
    
    % calculate R-squared
    r_squared = 1 - (mse / var_actual);

    %Mean absolute error
    mae = mean(abs(actual_vals - a_preds));
end