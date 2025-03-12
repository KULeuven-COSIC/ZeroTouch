close all;
clear all;
clc;

% Folder containing the .mat files
folderPath = '..\Measurements\Plotting Data\';

% Get a list of all .mat files starting with "recon"
matFiles = dir(fullfile(folderPath, 'recon*.mat'));

% Extract the numeric values from file names for sorting
numericValues = zeros(length(matFiles), 1); % Preallocate for efficiency
for i = 1:length(matFiles)
    % Extract the numeric part using regular expression
    tokens = regexp(matFiles(i).name, 'recon_error_(\d+\.\d+)', 'tokens');
    if ~isempty(tokens)
        numericValues(i) = str2double(tokens{1}); % Convert to a number
    end
end

% Sort the files based on the numeric values
[~, sortIdx] = sort(numericValues); % Get sorting indices
sortedMatFiles = matFiles(sortIdx); % Reorder files based on the sorting indices

% Loop through each sorted file and load it
allData = cell(length(sortedMatFiles), 1); % Preallocate for efficiency
for i = 1:length(sortedMatFiles)
    % Get the file name
    fileName = sortedMatFiles(i).name;
    fullFilePath = fullfile(folderPath, fileName);
    
    % Load the .mat file
    disp(['Loading: ', fullFilePath]); % Display the file being loaded
    allData{i} = load(fullFilePath); % Store the loaded data in the cell array
end

%% Reorganize the data

% Determine the number of files (cells) in allData
numFiles = length(allData); % Number of cells (files)

% Get the field names from the first structure in allData
fieldNames = fieldnames(allData{1});

% Assuming the first field contains the data you need
targetField = fieldNames{1}; % Choose the correct field (adjust if needed)

% Extract the actual arrays dynamically using the target field
extractedData = cellfun(@(x) x.(targetField), allData, 'UniformOutput', false);

%% Preprocess the data: Replace each array with the reshaped mean array
for i = 1:numFiles
    % Extract the original array from the current structure
    currentArray = extractedData{i}; 
    
    % Reshape the array to size (6, []) and transpose it
    reshapedArray = reshape(currentArray, 6, []).'; 
    
    % Compute the mean of the reshaped array along each column
    meanArray = mean(reshapedArray, 2); 
    
    % Replace the array in the structure with the new array
    extractedData{i} = meanArray; 
end

%% Reorganize the data into a matrix
% Determine the size of the arrays (assuming 1D arrays and they are equal-sized)
numElements = numel(extractedData{1}); % Number of elements in each array

% Preallocate a matrix to hold the reorganized data
reorganizedData = zeros(numElements, numFiles);

% Loop through each extracted array to reorganize data
for i = 1:numFiles
    currentArray = extractedData{i}; % Extract the numeric array from the cell
    reorganizedData(:, i) = currentArray(:); % Store as a column
end

%% Plots

threshold_used = 5.155552352039442;

% Number of plots corresponds to the number of rows in reorganizedData
numPlots = size(reorganizedData, 1);

% Extract the numeric values from the sorted file names for x-axis labels
xLabels = numericValues(sortIdx); % These correspond to the file names (already sorted)

% Loop through each row of reorganizedData to plot
for i = 1:102
    % Create a new figure for each plot
    figure;
    
    % Extract the current row
    currentRow = reorganizedData(i, :);
    
    % Plot the row (x-axis is simply the column indices)
    plot(currentRow, '-o'); % Keep the plot as before (1, 2, 3, ...)
    grid on; % Add grid for better readability
    yline(threshold_used, 'r', 'LineWidth', 1.2);

    % Add title and labels
    title(['Plot ', num2str(i)]);
    xlabel('Power Increases [dB]');
    ylabel('Reconstruction error');
    
    % Set x-axis ticks to match column indices (1, 2, 3, ...)
    xticks(1:length(xLabels)); % Match the number of columns in reorganizedData
    
    % Set x-axis labels to the numeric values from file names
    xticklabels(string(xLabels)); % Convert numeric labels to strings
end

%%

% reorganizedData_outside = reorganizedData(1:102,:);
% average_error = mean(reorganizedData_outside, 1);

% figure
% stem(average_error);
% xlabel('Power Increases [dB]');
% ylabel('Value');
% title('Average reconstruction errors');

% % Set x-axis ticks to match column indices (1, 2, 3, ...)
% xticks(1:length(xLabels)); % Match the number of columns in reorganizedData
% % Set x-axis labels to the numeric values from file names
% xticklabels(string(xLabels)); % Convert numeric labels to strings
