close all;
clear all;
clc

%%% ----------------------------
%%% Step 1: Load data
%%% ----------------------------

room_number = 1;
[inside_data, outside_data] = load_measurements(room_number);

% Generate coordinates for plotting

% first we load coordinates of the attackers, saved in a separate file
outside_coordinates = load("..\Measurements\Outside\outside_coordinates.txt");

% to translate close to the center of the coordinate system
% locations matter just for the plotting purposes
offset_x = 500;
offset_y = 710;

inside_coordinates = inside_data{1};
inside_coordinates(:,2) = inside_coordinates(:,2) - offset_x;
inside_coordinates(:,3) = inside_coordinates(:,3) - offset_y;

outside_coordinates(:,1) = outside_coordinates(:,1) - offset_x;
outside_coordinates(:,2) = outside_coordinates(:,2) - offset_y;

% Sort data and define attacker

inside_data_cropped = cell(1,length(inside_data));
outside_data_cropped = cell(1,102); % always 102 transmitters

for i=1:length(inside_data_cropped)
    % we just take the RSS values; they are located in the 6th column
    inside_data_cropped{i} = inside_data{i}(:,6);
end

attacker_increase = 7; % attacker would increase the transmit power, around 7 seems to be optimal for the attacker

% We generate the "attacker's" data
for i=1:length(outside_data_cropped)
    outside_data_cropped{i} = outside_data{i}(:,6) + attacker_increase;
end

% merge data, first 105 inside, next 102 outside
data_merged = [inside_data_cropped outside_data_cropped];

%%% ----------------------------
%%% Step 2: Prepare training data
%%% ----------------------------

% Set random seed for reproducibility
% rng(8);

% Number of possible inside transmitters
num_inside = length(inside_data);

% Number of possible features (number of possible receivers)
num_features = length(data_merged{1});

% For selecting the subset of features for training, for example if we only
% want to use 10 receivers, when this is set to 0 it uses all 105 of them
use_subset = 1;

% features_to_use is an array containing indices of features (receivers) used

% % good_feature_candidates = [1:13, 26, 39, 52, 65, 78, 85, 92, 95:99, 88, 81, 73, 104, 105, 66:70, 71, 53, 40, 27, 14]; % Room 1 Edge
% % % good_feature_candidates = [1:5, 10, 15, 20, 25, 30, 35, 36:40, 6, 11, 16, 21, 26, 31]; % Room 2 edge
% good_feature_candidates = [1:10, 20, 30, 29, 28, 38, 46, 54, 62, 70:78, 63, 55, 47, 39, 31, 21, 11]; % Room 3 edge
% % good_feature_candidates = good_feature_candidates(1:2:end);
% % % randomIndices = randperm(length(good_feature_candidates), length(good_feature_candidates));
% randomIndices = randperm(length(good_feature_candidates), 20);
% features_to_use = good_feature_candidates(randomIndices);

% random feature selection (region - 105, region 2 - 40, region 3 - 78)
% features_to_use = [1 3 5 10 15 16 17 22 30 36 43 46 51 59 69 73 74 83 87 96];
features_to_use = [7 13 17 19 22 24 35 39 41 57 62 66 69 73 74 77 78 81 92 100];
% features_to_use = randperm(num_features, 78);
% features_to_use = randperm(num_features, 70);
% features_to_use = randperm(num_features, 30);
% features_to_use = randperm(num_features, 25);

% Draw the localisation environment (walls of the room and the all transmitters and receivers)
draw_figure2(inside_coordinates, outside_coordinates, offset_x, offset_y, 0, [], [], features_to_use);

% Initialize training data matrix
X_train = zeros(num_features, num_inside);

% Initialize training data matrix (part)
X_train_part = zeros(length(features_to_use), num_inside);

% Loop through the first 105 cells to build X_train and X_train_part
for i = 1:num_inside
    % Extract RSS measurements and store as a column vector
    X_train(:, i) = data_merged{i}; % each cell is a column matrix

    % Extract RSS measurements and store as a column vector
    X_train_part(:, i) = data_merged{i}(features_to_use); % we only use receivers defined by features_to_use
end

var_per_receiver = var(X_train, 0, 2);
var_per_receiver_part = var(X_train_part(:, features_to_use), 0, 2);

[~, indices] = sort(var_per_receiver, 'descend');
[~, indices_part] = sort(var_per_receiver_part, 'descend');

%%% ----------------------------
%%% Step 3: Noise Addition to Training Data
%%% ----------------------------

% Define noise level and number of augmentations
SNR_dB = 20;
num_augmentations = 5; % Number of noisy samples per original sample

% Augment X_train with noise
X_train_noisy = [];
for i = 1:size(X_train, 2)
    original_sample = X_train(:, i);
    X_train_noisy = [X_train_noisy, original_sample]; % Include original sample
    for j = 1:num_augmentations
        noisy_sample = awgn(original_sample, SNR_dB, 'measured');
        X_train_noisy = [X_train_noisy, noisy_sample]; % add the noisy sample to the array
    end
end

% Augment X_train_part with noise
X_train_part_noisy = [];
for i = 1:size(X_train_part, 2)
    original_sample = X_train_part(:, i);
    X_train_part_noisy = [X_train_part_noisy, original_sample]; % Include original sample
    for j = 1:num_augmentations
        noisy_sample = awgn(original_sample, SNR_dB, 'measured');
        X_train_part_noisy = [X_train_part_noisy, noisy_sample]; % add the noisy sample to the array
    end
end

% Update the variables to include augmented data
X_train = X_train_noisy;
X_train_part = X_train_part_noisy;

%%% ----------------------------
%%% Step 4: Normalize Training Data
%%% ----------------------------

% Normalize the training data to have zero mean and unit variance
% Compute mu and sigma from the train signal
[X_train_norm, mu, sigma] = zscore(X_train, 0, 2);
[X_train_norm_part, mu_part, sigma_part] = zscore(X_train_part, 0, 2);

%%% ----------------------------
%%% Step 5: Train the Autoencoder
%%% ----------------------------

% Define the size of the hidden layer (number of neurons)
hiddenSize = 15;

if use_subset == 0
    % we use the whole set
    X_for_training = X_train_norm;
else
    % We can only use Rx data from the inside transceivers we are using
    selection_array = [];
    for k=1:length(features_to_use)
        selection_array = [selection_array features_to_use(k):(features_to_use(k)+num_augmentations)];
        % total length should be (num_augmentations + 1) * length(features_to_use)
    end
    X_for_training = X_train_norm_part(:, selection_array);
end

% Train the autoencoder
% Have samples as columns (required by trainAutoencoder)
autoenc = trainAutoencoder(X_for_training, hiddenSize, ...
    'MaxEpochs', 1500, ...
    'EncoderTransferFunction', 'logsig', ...
    'DecoderTransferFunction', 'logsig', ...
    'L2WeightRegularization', 0.01, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.01, ...
    'ScaleData', false);

%%% ----------------------------
%%% Step 6: Reconstruct Training Data and Compute Reconstruction Errors
%%% ----------------------------

% Reconstruct the training data
X_train_recon = predict(autoenc, X_for_training);

% Compute reconstruction error for training data
reconErrorTrain = sqrt(sum((X_for_training - X_train_recon).^2));

%%% ----------------------------
%%% Step 7: Determine the Threshold for Anomaly Detection
%%% ----------------------------

% Set the threshold as mean + k*std of the training reconstruction errors,
% this is non-optimal, a "static" way of determining treshold

threshold_mean = mean(reconErrorTrain) + 2*std(reconErrorTrain); % k = 2

threshold_max = max(reconErrorTrain);

threshold_median = median(reconErrorTrain);

threshold_percentile = prctile(reconErrorTrain, 95);

pd = fitdist(reconErrorTrain', 'Normal');
alpha = 0.01;
threshold_distribution = icdf(pd, 1 - alpha);

mu = mean(reconErrorTrain);
sigma = std(reconErrorTrain);
% Desired maximum proportion of anomalies
p = 0.05;
% Calculate k using Chebyshev's inequality
k = sqrt(1 / p);
% Set the threshold
threshold_z_score = mu + k * sigma;

% Calculate the first and third quartiles
Q1 = prctile(reconErrorTrain, 25);
Q3 = prctile(reconErrorTrain, 75);
% Calculate IQR
IQR = Q3 - Q1;
% Set the threshold
threshold_IQR = Q3 + 1.5 * IQR; % 1.5 is a common multiplier

% Select a high threshold (e.g., 90th percentile)
high_threshold = prctile(reconErrorTrain, 90);
% Extract excesses over the threshold
excesses = reconErrorTrain(reconErrorTrain > high_threshold) - high_threshold;
% Fit GPD to the excesses
pd = fitdist(excesses', 'GeneralizedPareto');
% Set desired exceedance probability (e.g., 0.01 for 1%)
p_exceedance = 0.01;
% Calculate the threshold using the GPD
threshold_EVT = high_threshold + icdf(pd, 1 - p_exceedance);

% Fit a GMM with two components (normal and anomalies)
gm = fitgmdist(reconErrorTrain', 2);
% Identify the component with the lower mean (assumed to be normal data)
[~, idx] = min(gm.mu);
normal_component = idx;
% Calculate the threshold as the mean plus a multiple of the standard deviation of the normal component
mu_normal = gm.mu(normal_component);
sigma_normal = sqrt(gm.Sigma(normal_component));
threshold_gaussian = mu_normal + 3 * sigma_normal; % Adjust multiplier as needed

%%

flag_th = (threshold_max - threshold_distribution >= 0.1) && (abs(threshold_max - threshold_EVT) <= 0.3) && (abs(threshold_distribution - threshold_IQR) >= 0.1);
flag_th_2 = threshold_IQR - threshold_max < 0.1;

threshold = threshold_IQR;

%%% ----------------------------
%%% Step 8: Prepare Test Data (Outside Transmitters)
%%% ----------------------------

% Number of outside transmitters
num_outside = 102;

% Initialize test data matrix
X_test = zeros(num_features, num_outside);
X_test_part = zeros(length(features_to_use), num_outside); % We need to use only subset of all features

% Loop through cells 106 to 207 to build X_test and X_test_part
for i = 1:num_outside
    % Index in data_merged
    idx = num_inside + i;
    % Extract RSS measurements and store as a column vector
    X_test(:, i) = data_merged{idx};
    X_test_part(:, i) = data_merged{idx}(features_to_use);
end

% Augment X_test with noise
X_test_noisy = [];
for i = 1:size(X_test, 2)
    original_sample = X_test(:, i);
    X_test_noisy = [X_test_noisy, original_sample]; % Include original sample
    for j = 1:num_augmentations
        noisy_sample = awgn(original_sample, SNR_dB, 'measured');
        X_test_noisy = [X_test_noisy, noisy_sample]; % add the noisy sample to the array
    end
end

% Augment X_test_part with noise
X_test_part_noisy = [];
for i = 1:size(X_test_part, 2)
    original_sample = X_test_part(:, i);
    X_test_part_noisy = [X_test_part_noisy, original_sample]; % Include original sample
    for j = 1:num_augmentations
        noisy_sample = awgn(original_sample, SNR_dB, 'measured');
        X_test_part_noisy = [X_test_part_noisy, noisy_sample]; % add the noisy sample to the array
    end
end

% Update the variables to include augmented data
X_test = X_test_noisy;
X_test_part = X_test_part_noisy;

% Normalize the test data using the same mean and standard deviation as training data
% [X_test_norm, mu, sigma] = zscore(X_test, 0, 2);
% [X_test_norm_part, mu_part, sigma_part] = zscore(X_test_part, 0, 2);

X_test_norm = (X_test - mu) ./ sigma;
X_test_norm_part = (X_test_part - mu_part) ./ sigma_part;

%%% ----------------------------
%%% Step 9: Reconstruct Test Data and Compute Reconstruction Errors
%%% ----------------------------

% Expand the test data
% here we add the remaining inside transceivers we did not use for
% training, we can say they are devices that want to join the network

selection_array = [];
if use_subset == 0
    X_for_reconstruction = X_test_norm; % we add nothing, we are using all transceivers for training
else
    features_not_used_Tx = setdiff(1:num_features,features_to_use); % which features are we NOT using for traning
    features_not_used_Tx_exapnded = [];
    for i=1:length(features_not_used_Tx)
        features_not_used_Tx_exapnded = [features_not_used_Tx_exapnded features_not_used_Tx(i) + (i-1)*num_augmentations]; % positions in the noisy array
    end

    for k=1:length(features_not_used_Tx_exapnded)
        selection_array = [selection_array features_not_used_Tx_exapnded(k):(features_not_used_Tx_exapnded(k)+num_augmentations)];
    end

    features_not_used_act_as_tx = X_train_norm(features_to_use, selection_array); % select features NOT used for traning
    X_for_reconstruction = [X_test_norm_part features_not_used_act_as_tx]; % include these features to the test array
end

X_test_recon = predict(autoenc, X_for_reconstruction);

% Compute reconstruction error for test data
reconErrorTest = sqrt(sum((X_for_reconstruction - X_test_recon).^2));
errors_per_receiver =  reshape(reconErrorTest, num_augmentations + 1, []).';
errors_per_receiver_averaged = mean(errors_per_receiver, 2);

%%% ----------------------------
%%% Step 10: Classify Test Data Based on Reconstruction Error
%%% ----------------------------

% Classify as anomaly if reconstruction error exceeds threshold

anomaly_labels_all = reconErrorTest > threshold; % 1 for anomalies (outside transmitters), 0 for normal (inside transmitters)

sub_arrays =  reshape(anomaly_labels_all, num_augmentations + 1, []).';

anomaly_labels = [];
for i=1:length(sub_arrays(:,1))
    if sum(sub_arrays(i,:)) >= int16(num_augmentations/2) + 1
        anomaly_labels = [anomaly_labels 1];
    else
        anomaly_labels = [anomaly_labels 0];
    end
end

anomaly_labels = anomaly_labels_all; % if we want to test all samples

% Since we have augmented data, replicate the true labels accordingly
if use_subset == 0
    % All test data are outside transmitters (anomalies)
    true_labels = ones(1, num_outside); % 1 indicates outside transmitters
else
    % First we have outsite set (all anomalies == ones), then we have
    % inside test, regular situation
    true_labels = [ones(1, num_outside*(num_augmentations+1)) zeros(1, length(features_not_used_Tx)*(num_augmentations+1))]; % when we test all samples
    % true_labels = [ones(1, num_outside) zeros(1, length(features_not_used_Tx))]; % only features tested
end

%%% ----------------------------
%%% Step 11: Evaluate Performance
%%% ----------------------------

% Calculate confusion matrix
[c, cm] = confusion(true_labels, double(anomaly_labels));

% Display confusion matrix
figure;
plotconfusion(true_labels, anomaly_labels);
title('Confusion Matrix');

% Calculate and display performance metrics
accuracy = 100 * (1 - c);
fprintf('Test Set Accuracy: %.2f%%\n', accuracy);

% Compute other metrics
tp = cm(2,2); % True positives
tn = cm(1,1); % True negatives
fp = cm(1,2); % False positives
fn = cm(2,1); % False negatives

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

%%% ----------------------------
%%% Step 12: Visualize Reconstruction Errors
%%% ----------------------------

% Plot histogram of reconstruction errors
figure;
histogram(reconErrorTest, 20);
hold on;
xline(threshold, 'r', 'LineWidth', 2);
xlabel('Reconstruction Error');
ylabel('Frequency');
title('Reconstruction Error Distribution for Test Data');
legend('Reconstruction Error', 'Threshold');

% ----------------------------
% Visualize Training Reconstruction Errors
% ----------------------------

% Plot histogram of training reconstruction errors
figure;
histogram(reconErrorTrain, 20);
hold on;
xline(threshold, 'r', 'LineWidth', 2);
xlabel('Reconstruction Error');
ylabel('Frequency');
title('Reconstruction Error Distribution for Training Data');
legend('Reconstruction Error', 'Threshold');

%%% ----------------------------
%%% Step 13: Perform PCA Analysis on Training Data
%%% ----------------------------
perfrom_pca = 0;

if perfrom_pca == 1
% Transpose training data to have samples as rows and features as columns
X_train_T_norm = X_train_norm';

% Perform PCA on normalized training data
[coeff, score_train, latent, tsquared, explained, mu_pca] = pca(X_train_T_norm);

X_test_T_norm = X_test_norm';

% Project test data onto principal components
score_test = (X_test_T_norm - mu_pca) * coeff;

% Create labels for visualization
labels_train = zeros(size(X_train_T_norm, 1), 1); % 0 for inside transmitters (normal)
labels_test = ones(size(X_test_T_norm, 1), 1);    % 1 for outside transmitters (anomalies)
labels_combined = [labels_train; labels_test];

% Combine scores for visualization
score_combined = [score_train; score_test];

% Plot cumulative variance explained
figure;
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('PCA - Variance Explained by Principal Components (Training Data)');

% Determine number of components to retain
cumulativeVariance = cumsum(explained);
numComponents_95 = find(cumulativeVariance >= 95, 1);
fprintf('\nPCA Analysis\n');
fprintf('Number of principal components to retain for 95%% variance: %d\n', numComponents_95);

numComponents_85 = find(cumulativeVariance >= 85, 1);
fprintf('Number of principal components to retain for 85%% variance: %d\n', numComponents_85);

numComponents_75 = find(cumulativeVariance >= 75, 1);
fprintf('Number of principal components to retain for 75%% variance: %d\n', numComponents_75);

numComponents_70 = find(cumulativeVariance >= 70, 1);
fprintf('Number of principal components to retain for 70%% variance: %d\n', numComponents_70);

% Analyze feature contributions to the first principal component
pc1_coeff = coeff(:,1);
[sorted_coeff, feature_indices] = sort(abs(pc1_coeff), 'descend');
top_features = feature_indices(1:10);
disp('Top 10 features contributing to PC1:');
disp(top_features');

% Plot the loadings of the first principal component
figure;
bar(pc1_coeff);
xlabel('Feature Index');
ylabel('Loading');
title('Feature Loadings on Principal Component 1');

% Visualize data in the space of the first two principal components
figure;
gscatter(score_combined(:,1), score_combined(:,2), labels_combined, 'br', 'ox');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('Data Projected onto First Two Principal Components');
legend('Inside Transmitters (Normal)', 'Outside Transmitters (Anomalies)');
end


%%% ----------------------------
%%% Step 14: Generate ROC Curve
%%% ----------------------------

fprintf('ROC Curve analysis\n');

if use_subset == 0
    % Use all features
    % Combine normalized training and test data
    X_total_norm = [X_train_norm, X_test_norm];
    
    % Combine true labels
    true_labels_total = [zeros(1, num_inside), ones(1, num_outside)]; % 0 for normal (training data), 1 for anomalies (test data)
    
    % Reconstruct all data
    X_total_recon = predict(autoenc, X_total_norm);
    
    % Compute reconstruction errors
    reconErrorTotal = sqrt(sum((X_total_norm - X_total_recon).^2));
    
    % Reconstruction errors for test data
    X_test_recon = predict(autoenc, X_test_norm);
    reconErrorTest = sqrt(sum((X_test_norm - X_test_recon).^2));
    
    % True labels for test data
    true_labels_test = ones(1, num_outside); % All test data are anomalies
else
    % Use only the subset of features
    % Combine test anomalies and additional inside measurements
    X_test_total_norm = [X_test_norm_part, features_not_used_act_as_tx];
    
    % Combine normalized training data and test data
    X_total_norm = [X_for_training, X_test_total_norm];

    % Combine true labels
    % true_labels_total = [zeros(1, length(features_to_use)), ones(1, num_outside), zeros(1, length(features_not_used_Tx))];
    true_labels_total = [zeros(1, length(features_to_use)*(num_augmentations+1)), ones(1, num_outside*(num_augmentations+1)), zeros(1, length(features_not_used_Tx)*(num_augmentations+1))];

    % Reconstruct all data using the trained autoencoder
    X_total_recon = predict(autoenc, X_total_norm);
    
    % Compute reconstruction errors for all data
    reconErrorTotal = sqrt(sum((X_total_norm - X_total_recon).^2));
    
    % Reconstruction errors for test data
    X_test_total_recon = predict(autoenc, X_test_total_norm);
    reconErrorTest = sqrt(sum((X_test_total_norm - X_test_total_recon).^2));
    
    % True labels for test data
    true_labels_test_o = [ones(1, num_outside), zeros(1, length(features_not_used_Tx))];
    true_labels_test = [ones(1, num_outside*(num_augmentations+1)), zeros(1, length(features_not_used_Tx)*(num_augmentations+1))];

end

% Define thresholds
minError = min(reconErrorTotal);
maxError = max(reconErrorTotal);
numThresholds = 2001;
thresholds = linspace(maxError, minError, numThresholds - 1);

% add original threshold to the array of thresholds
thresholds = sort([thresholds, threshold], 'descend'); % now indeed we have 2001

% Initialize TPR and FPR arrays
TPR = zeros(1, numThresholds);
FPR = zeros(1, numThresholds);

for i = 1:numThresholds
    % Classify based on threshold
    anomaly_labels_total = reconErrorTotal > thresholds(i);
    
    % Compute confusion matrix components
    TP = sum((anomaly_labels_total == 1) & (true_labels_total == 1));
    TN = sum((anomaly_labels_total == 0) & (true_labels_total == 0));
    FP = sum((anomaly_labels_total == 1) & (true_labels_total == 0));
    FN = sum((anomaly_labels_total == 0) & (true_labels_total == 1));
    
    % Calculate TPR and FPR
    TPR(i) = TP / (TP + FN);
    FPR(i) = FP / (FP + TN);
end

% Plot ROC curve
figure;
plot(FPR, TPR, '-b', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;
hold on;
plot([0,1], [0,1], '--r');
legend('ROC Curve', 'Random Classifier');

% Calculate AUC
AUC = trapz(FPR, TPR);
fprintf('Area Under the ROC Curve (AUC): %.4f\n', AUC);

% Find optimal threshold
youdenJ = TPR - FPR;
[~, optimalIdx] = max(youdenJ);
optimalThreshold = thresholds(optimalIdx);
fprintf('Optimal Threshold based on Youden''s J statistic: %.4f\n\n', optimalThreshold);

% Plot optimal point
plot(FPR(optimalIdx), TPR(optimalIdx), 'og', 'MarkerSize', 10, 'LineWidth', 2);
legend('ROC Curve', 'Random Classifier', 'Optimal Threshold');

%%% ----------------------------
%%% Step 15: Evaluate with Optimal Threshold
%%% ---------------------------

% Classify test data with optimal threshold
anomaly_labels_optimal = reconErrorTest > optimalThreshold;
anomaly_labels_temp = reconErrorTest > threshold; % use non-optimal thershold

% Compute confusion matrix for test data
[~, cm_optimal] = confusion(true_labels_test, anomaly_labels_optimal);

% Display confusion matrix
% figure;
% plotconfusion(true_labels_test, anomaly_labels_optimal);
% title('Confusion Matrix with Optimal Threshold');

% Majority voting strategy confusion matrix
subarrays = reshape(anomaly_labels_optimal, num_augmentations + 1, []).';
anomaly_labels_optimal_o = [];
for i=1:length(subarrays(:,1))
    if sum(subarrays(i,:)) >= int16((num_augmentations + 1)/2)
        anomaly_labels_optimal_o = [anomaly_labels_optimal_o 1];
    else
        anomaly_labels_optimal_o = [anomaly_labels_optimal_o 0];
    end
end

% Display confusion matrix
figure;
plotconfusion(true_labels_test_o, anomaly_labels_optimal_o);
title('Confusion Matrix with Optimal Threshold + majority voting');

% Majority voting strategy confusion matrix without optimal thershld
subarrays = reshape(anomaly_labels_temp, num_augmentations + 1, []).';
anomaly_labels_o = [];
for i=1:length(subarrays(:,1))
    if sum(subarrays(i,:)) >= int16((num_augmentations + 1)/2)
        anomaly_labels_o = [anomaly_labels_o 1];
    else
        anomaly_labels_o = [anomaly_labels_o 0];
    end
end

% Display confusion matrix
figure;
plotconfusion(true_labels_test_o, anomaly_labels_o);
title('Confusion Matrix with majority voting');

% Averaged errors per sample confusion matrix
anomaly_labels_optimal_a = errors_per_receiver_averaged > optimalThreshold;

% Compute confusion matrix for test data
[~, cm_optimal] = confusion(true_labels_test_o, anomaly_labels_optimal_a');

% Display confusion matrix
% figure;
% plotconfusion(true_labels_test_o, anomaly_labels_optimal_a');
% title('Confusion Matrix with averaged errors');

% draw the localisation environment again now indicating which
% transmitterts and receivers are wrongly classified
wrong_position = xor(true_labels_test_o, anomaly_labels_optimal_o)';
wrong_position_outside = wrong_position(1:102);
wrong_position_inside = wrong_position(103:end);

wrong_position_outside_number = 1:102;
wrong_position_outside_number = wrong_position_outside_number(wrong_position_outside == 1);
wrong_position_inside_number = features_not_used_Tx(wrong_position_inside == 1);

draw_figure2(inside_coordinates, outside_coordinates, offset_x, offset_y, 1, wrong_position_inside_number, wrong_position_outside_number, features_to_use);

% For non optimal threshold, visualisation
wrong_position = xor(true_labels_test_o, anomaly_labels_o)';
wrong_position_outside = wrong_position(1:102);
wrong_position_inside = wrong_position(103:end);

wrong_position_outside_number = 1:102;
wrong_position_outside_number = wrong_position_outside_number(wrong_position_outside == 1);
wrong_position_inside_number = features_not_used_Tx(wrong_position_inside == 1);
draw_figure2(inside_coordinates, outside_coordinates, offset_x, offset_y, 1, wrong_position_inside_number, wrong_position_outside_number, features_to_use);

%%

errors_reshaped_test = reshape(reconErrorTest, num_augmentations + 1, []).';
average_errors_test = mean(errors_reshaped_test,2);

errors_reshaped_train = reshape(reconErrorTrain, num_augmentations + 1, []).';
average_errors_train = mean(errors_reshaped_train,2);
% labels = [ones(102,1)' zeros(num_inside - length(features_to_use),1)'];
threshold_min = min(reconErrorTrain);

figure;
hold on;

stem(reconErrorTrain, 'Marker', 'o', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2); % Black/gray for the stem plot

% Add a horizontal line for the thresholds
yline(optimalThreshold, 'b', 'LineWidth', 1.2);
yline(threshold_min, 'r', 'LineWidth', 1.2);
yline(threshold_EVT, 'm', 'LineWidth', 1.2);
yline(threshold_IQR, 'k', 'LineWidth', 1.2);
yline(threshold_max, 'c', 'LineWidth', 1.2);
yline(threshold_median, 'w', 'LineWidth', 1.3);
yline(threshold_mean, 'r--', 'LineWidth', 1.2);
yline(threshold_z_score, 'k--', 'LineWidth', 1.2);
yline(threshold_percentile, 'g', 'LineWidth', 1.2);
yline(threshold_distribution, 'y', 'LineWidth', 1.2);
yline(threshold_gaussian, 'b--', 'LineWidth', 1.2);

% Add legend
hLegend = legend('Error per transmitter', 'Optimal Threshold', 'Min error in training', 'Extreme Value Theory (EVT)', 'Interquartile Range (IQR) Method', 'Max error in training', 'Median error in training', 'Mean error in training','Z-Score', 'Percentile', 'Parametric Statistical Modeling', 'Gaussian Mixture Model (GMM)', 'Location', 'best');
title('Stem Plot of Average Errors of a Train set per sample');
xlabel('Sample Index');
ylabel('Average Error');

set(hLegend, 'Color', [0.91 0.91 0.91]); % Light gray background

hold off;

figure;
hold on;

stem(average_errors_train, 'Marker', 'o', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2); % Black/gray for the stem plot

% Add a horizontal line for the thresholds
yline(optimalThreshold, 'b', 'LineWidth', 1.2);
yline(threshold_min, 'r', 'LineWidth', 1.2);
yline(threshold_EVT, 'm', 'LineWidth', 1.2);
yline(threshold_IQR, 'k', 'LineWidth', 1.2);
yline(threshold_max, 'c', 'LineWidth', 1.2);
yline(threshold_median, 'w', 'LineWidth', 1.3);
yline(threshold_mean, 'r--', 'LineWidth', 1.2);
yline(threshold_z_score, 'k--', 'LineWidth', 1.2);
yline(threshold_percentile, 'g', 'LineWidth', 1.2);
yline(threshold_distribution, 'y', 'LineWidth', 1.2);
yline(threshold_gaussian, 'b--', 'LineWidth', 1.2);

% Add legend
hLegend = legend('Error per transmitter', 'Optimal Threshold', 'Min error in training', 'Extreme Value Theory (EVT)', 'Interquartile Range (IQR) Method', 'Max error in training', 'Median error in training', 'Mean error in training','Z-Score', 'Percentile', 'Parametric Statistical Modeling', 'Gaussian Mixture Model (GMM)', 'Location', 'best');
title('Stem Plot of Average Errors of a Train set per receiver');
xlabel('Receiver Index');
ylabel('Average Error');

set(hLegend, 'Color', [0.91 0.91 0.91]); % Light gray background

hold off;

figure;
hold on;

% Plotting average errors with a stem plot in black/gray color
stem(average_errors_test, 'Marker', 'o', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2);

% Add a horizontal line for the thresholds
yline(optimalThreshold, 'b', 'LineWidth', 1.2);
yline(threshold_min, 'r', 'LineWidth', 1.2);
yline(threshold_EVT, 'm', 'LineWidth', 1.2);
yline(threshold_IQR, 'k', 'LineWidth', 1.2);
yline(threshold_max, 'c', 'LineWidth', 1.2);
yline(threshold_median, 'w', 'LineWidth', 1.3);
yline(threshold_mean, 'r--', 'LineWidth', 1.2);
yline(threshold_z_score, 'k--', 'LineWidth', 1.2);
yline(threshold_percentile, 'g', 'LineWidth', 1.2);
yline(threshold_distribution, 'y', 'LineWidth', 1.2);
yline(threshold_gaussian, 'b--', 'LineWidth', 1.2);


% Shade the region corresponding to the first 102 values (for anomalies or a specific case)
area(1:102, max(average_errors_test) * ones(1, 102), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [1 0 0]); % Red for anomalies
area(103:length(average_errors_test), max(average_errors_test) * ones(1, length(average_errors_test)-102), ...
    'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [0 0.5 0]); % Darker green for normal data

% Add text labels to indicate regions with matching colors
text(20, max(average_errors_test) * 0.95, 'Anomalies', 'FontSize', 12, 'Color', [1 0 0], 'HorizontalAlignment', 'center');
text(115, max(average_errors_test) * 0.95, 'Normal', 'FontSize', 12, 'Color', [0 0.5 0], 'HorizontalAlignment', 'center');

% Add legend
hLegend = legend('Error per transmitter', 'Optimal Threshold', 'Min error in training', 'Extreme Value Theory (EVT)', 'Interquartile Range (IQR) Method', 'Max error in training', 'Median error in training', 'Mean error in training','Z-Score', 'Percentile', 'Parametric Statistical Modeling', 'Gaussian Mixture Model (GMM)', 'Location', 'best');
title('Stem Plot of Average Errors of a Test set');
xlabel('Transmitter Index');
ylabel('Average Error');

set(hLegend, 'Color', [0.91 0.91 0.91]); % Light gray background

hold off;


%%

figure('Position', [100, 100, 1240, 930]); % [x, y, width, height]
% figure
hold on;

average_errors_test_temp = average_errors_test;

% Plotting average errors with a stem plot in black/gray color
ylim([0, max(average_errors_test_temp)]); % Add a small buffer if needed, or adjust as per your preference
xlim([1, length(average_errors_test_temp)]);
stem(average_errors_test_temp, 'Marker', 'o', 'Color', [0 0 0], 'LineWidth', 1.2);

% Add a horizontal line for the thresholds
% yline(optimalThreshold, 'b', 'LineWidth', 1.2);
% yline(threshold_min, 'r', 'LineWidth', 1.2);
% yline(threshold_EVT, 'm', 'LineWidth', 1.2);
yline(threshold_IQR, 'b', 'LineWidth', 1.4);
% yline(threshold_max, 'c', 'LineWidth', 1.2);
% yline(threshold_median, 'w', 'LineWidth', 1.3);
% yline(threshold_mean, 'r--', 'LineWidth', 1.2);
% yline(threshold_z_score, 'k--', 'LineWidth', 1.2);
% yline(threshold_percentile, 'g', 'LineWidth', 1.2);
% yline(threshold_distribution, 'y', 'LineWidth', 1.2);
% yline(threshold_gaussian, 'b--', 'LineWidth', 1.2);


% Shade the region corresponding to the first 102 values (for anomalies or a specific case)
area(1:102, max(average_errors_test_temp) * ones(1, 102), 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [1 0 0]); % Red for anomalies

% Define the range for normal data
max_val = max(average_errors_test_temp);
x = 1:length(average_errors_test_temp);

% Parameters for horizontal shading
stripe_height = 1; % Height of each stripe

% Loop through to create horizontal stripes
for y = 0:2 * stripe_height:max_val
    % Define vertices of the stripe (rectangle)
    x_coords = [103, length(x), length(x), 103]; % Full width of the "Normal" region
    y_coords = [y, y, y + stripe_height, y + stripe_height];

    % Ensure the stripe stays within the max value
    y_coords(y_coords > max_val) = max_val;

    % Plot the horizontal stripe
    fill(x_coords, y_coords, [0 0.5 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Dark green
end


% area(103:length(average_errors_test_temp), max(average_errors_test_temp) * ones(1, length(average_errors_test_temp)-102), ...
%     'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [0 0.5 0]); % Darker green for normal data

% Add text labels to indicate regions with matching colors
% text(20, max(average_errors_test_temp) * 0.9, 'Anomalies', 'FontSize', 17, 'Color', [1 0 0], 'HorizontalAlignment', 'center');
% text(115, max(average_errors_test_temp) * 0.9, 'Normal', 'FontSize', 17, 'Color', [0 0.5 0], 'HorizontalAlignment', 'center');

% Add legend
% hLegend = legend('Error per transmitter', 'Interquartile Range (IQR) Method Threshold', 'Location', 'best');
% title('Stem Plot of Average Errors of a Test set');
xlabel('Transmitter Index');
ylabel('Average Error');

ax = gca; 
ax.FontSize = 19; 

% set(hLegend, 'Color', [1 1 1]); % Light gray background
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white');


hold off;
%%

% for testing

testing_labels = true_labels_test_o;

testing_labels(1:8) = 0;
testing_labels(118:119) = 1;

figure;
plotconfusion(true_labels_test_o, testing_labels); % (targets,outputs)
ax = gca; % Get current axes
textHandles = findall(ax, 'Type', 'Text'); % Find all text objects
set(textHandles, 'FontSize', 18);
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white');