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
outside_data_cropped = cell(1,102);

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
%% Step 2: Prepare training data
%%% ----------------------------

% Number of possible inside transmitters
num_inside = length(inside_data);

% Number of outside transmitters
num_outside = 102;

% Total number of samples (one device measures inside + outside devices)
num_samples = num_inside + num_outside;

% Number of possible features (number of possible receivers, these are features)
num_features = length(data_merged{1});

% Initialize data matrix and labels
X = zeros(num_inside, num_samples); % Features
y = zeros(1, num_samples); % Labels

% Prepare data and labels
for i = 1:num_samples
    % Extract RSS measurements and store as column vectors
    X(:, i) = data_merged{i}; % All features

    if i <= num_inside
        % Inside measurements
        y(i) = 0; % Label for inside
    else
        % Outside measurements
        y(i) = 1; % Label for outside
    end
end

receivers_to_use = 1:length(inside_data);

use_subset = 1;

% Set random seed for reproducibility
% rng(1);

if use_subset == 1
    % Crop data, how many receivers are we using
    receivers_to_use = [7 13 17 19 22 24 35 39 41 57 62 66 69 73 74 77 78 81 92 100];
    % receivers_to_use = randperm(num_inside, 20);
    X_part = X(receivers_to_use, :);
    X = X_part;
end


%% Shuffle and Split the Data into Training and Test Sets

% Create a random permutation of indices
rand_indices = randperm(num_samples);

% Shuffle the data
X_shuffled = X(:, rand_indices);
y_shuffled = y(rand_indices);

% Define the proportion of data to be used for training
train_ratio = 1;

% Number of training samples
num_train = round(train_ratio * num_samples);

% Split the data
X_train = X_shuffled(:, 1:num_train); % number of rows says how many receivers are we using
y_train = y_shuffled(1:num_train);
points_for_training = rand_indices(1:num_train); % says which transmitters are we using

% % Idealized scenario
X_test = X_train;
y_test = y_train;
points_for_testing = points_for_training;

% X_test = X_shuffled(:, num_train+1:end);
% y_test = y_shuffled(num_train+1:end);
% points_for_testing = rand_indices(num_train+1:end);

draw_figure3(inside_coordinates, outside_coordinates, offset_x, offset_y, 0, [], points_for_training, points_for_testing, receivers_to_use);

%% Noise Addition

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

y_train_noisy = repmat(y_train, num_augmentations + 1, 1);
y_train_noisy = y_train_noisy(:)';

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

y_test_noisy = repmat(y_test, num_augmentations + 1, 1);
y_test_noisy = y_test_noisy(:)';

% Transform the data

% X_train_noisy_transformed = histogram_transform(X_train_noisy, [], 1);
% X_test_noisy_transformed = histogram_transform(X_train_noisy, X_test_noisy, 0);

% X_train_noisy_transformed = histogram_transform_v2(X_train_noisy, [], 1);
% X_test_noisy_transformed = histogram_transform_v2(X_train_noisy, X_test_noisy, 0);

%% Normalize the Data

% Compute mu and sigma from the training data only
[~, mu, sigma] = zscore(X_train_noisy, 0, 2);

% Normalize the training and test data using mu and sigma from training data
X_train_norm = (X_train_noisy - mu) ./ sigma;
X_test_norm = (X_test_noisy - mu) ./ sigma;

% Handle potential division by zero in normalization (if sigma is zero)
sigma_zero_indices = (sigma == 0);
X_train_norm(sigma_zero_indices, :) = 0;
X_test_norm(sigma_zero_indices, :) = 0;

%% Build and Train the Neural Network

% Transpose the input data to have samples as rows
% trainNetwork expects input data where samples are rows and features are columns.
X_train_T = X_train_norm';
y_train_T = y_train_noisy';

X_test_T = X_test_norm';
y_test_T = y_test_noisy';

% Convert numeric labels to categorical
y_train_cat = categorical(y_train_T);
y_test_cat = categorical(y_test_T);

% Define the Neural Network Architecture

inputSize = size(X_train_T, 2); % Number of input features
hiddenSize = 10; % Number of neurons in the hidden layer
numClasses = 2; % Number of classes (inside and outside)

% layers = [
%     featureInputLayer(inputSize, 'Normalization', 'none') % Input layer
%     fullyConnectedLayer(hiddenSize)                       % Hidden layer
%     reluLayer                                             % Activation function
%     dropoutLayer(0.2)
%     fullyConnectedLayer(numClasses)                       % Output layer with two neurons
%     softmaxLayer                                          % Softmax activation for probabilities
%     classificationLayer];                                 % Classification output layer

% layers = [
%     featureInputLayer(inputSize, 'Normalization', 'none') % Input layer
%     fullyConnectedLayer(15)                              % Hidden layer 1
%     reluLayer
%     fullyConnectedLayer(32)                              % Hidden layer 2
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(numClasses)                      % Output layer
%     softmaxLayer
%     classificationLayer];

layers = [
    featureInputLayer(inputSize, 'Normalization', 'none') % Input layer
    fullyConnectedLayer(15)                              % Hidden layer 1
    reluLayer
    fullyConnectedLayer(numClasses)                      % Output layer
    softmaxLayer
    classificationLayer];

% Visualize the Network Architecture
analyzeNetwork(layers);

% Specify Training Options

% options = trainingOptions('adam', ...
%     'MaxEpochs', 70, ...
%     'MiniBatchSize', 16, ...
%     'InitialLearnRate', 0.01, ...
%     'L2Regularization', 0.001, ...
%     'Shuffle', 'every-epoch', ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...       % Reduced learning rate
    'MaxEpochs', 200, ...                % Increased epochs
    'MiniBatchSize', 32, ...             % Adjusted batch size
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0001, ...      % Added L2 regularization
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the Neural Network

% Train the network
net = trainNetwork(X_train_T, y_train_cat, layers, options);

%% Evaluate the Model

% Predict on test data

% Get predicted class probabilities
y_pred_probs = predict(net, X_test_T);

% Get predicted labels
[~, y_pred_indices] = max(y_pred_probs, [], 2);
y_pred = y_pred_indices - 1; % Convert indices to labels (0 and 1)

% Convert categorical test labels to numeric
y_test_numeric = double(y_test_cat);
y_test_numeric = y_test_numeric - 1; % Convert categories to labels (0 and 1)

% Calculate performance metrics

% Accuracy
accuracy = sum(y_pred == y_test_numeric) / numel(y_test_numeric);
fprintf('Test Set Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
plotconfusion(logical(int16(y_test_numeric))', logical(int16(y_pred))');
title('Confusion Matrix');

% Compute other metrics
tp = sum((y_pred == 1) & (y_test_numeric == 1)); % True positives
tn = sum((y_pred == 0) & (y_test_numeric == 0)); % True negatives
fp = sum((y_pred == 1) & (y_test_numeric == 0)); % False positives
fn = sum((y_pred == 0) & (y_test_numeric == 1)); % False negatives

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 Score: %.2f\n', f1_score);

% Majority voting strategy confusion matrix
subarrays = reshape(y_pred, num_augmentations + 1, []).';
labels_optimal = [];
for i=1:length(subarrays(:,1))
    if sum(subarrays(i,:)) >= int16((num_augmentations + 1)/2)
        labels_optimal = [labels_optimal 1];
    else
        labels_optimal = [labels_optimal 0];
    end
end

figure;
plotconfusion(logical(int16(y_test)), labels_optimal);
title('Confusion Matrix - majority voting');

wrong_position = xor(logical(int16(y_test)), labels_optimal)';
wrong_position_transmitters = points_for_testing(wrong_position == 1);
draw_figure3(inside_coordinates, outside_coordinates, offset_x, offset_y, 1, wrong_position_transmitters, points_for_training, points_for_testing, receivers_to_use);
