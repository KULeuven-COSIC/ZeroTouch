function [inside_data, outside_data] = load_measurements(flag)

filePath_inside = '..\Measurements\Inside\';
filePath_outside = '..\Measurements\Outside\';

if flag == 1

% Load inside data
inside_data = cell(1,105);
for i=1:105
    fileName = sprintf('%sz1-p%d-rx1.txt', filePath_inside, i);
    inside_data{i} = load(fileName);
    % Cell one in the set represents transceiver 1
    % transmitting and all other transceivers receiving
    % that is why we have a total of 105 cells (105 inside transceivers)
    % and each cell has 105 rows
end

% Load outside data
% Data was saved in different .txt files, that is why we load it from parts

outside_data = cell(1,102);
for i=1:7
    data_index = i;
    fileName = sprintf('%stx1_%d_z1.txt', filePath_outside, i);
    % Cell one in the set represents transmitter 1 (outside) transmitting
    % and all inside transceivers receiving, that is why we have a total of
    % 105 rows in each cell (105 inside transceivers), and the outside_data
    % has 102 total cells (102 outside transmitters)
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx2_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:12
    data_index = data_index + 1;
    fileName = sprintf('%stx3_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx4_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx5_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:21
    data_index = data_index + 1;
    fileName = sprintf('%stx6_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:8
    data_index = data_index + 1;
    fileName = sprintf('%stx7_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:18
    data_index = data_index + 1;
    fileName = sprintf('%stx8_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx9_%d_z1.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

elseif flag == 2

% Load inside data
inside_data = cell(1,40);
for i=1:40
    fileName = sprintf('%sz2-p%d-rx2.txt', filePath_inside, i);
    inside_data{i} = load(fileName);
    % Cell one in the set represents transceiver 1
    % transmitting and all other transceivers receiving
    % that is why we have a total of 105 cells (105 inside transceivers)
    % and each cell has 105 rows
end

% Load outside data
% Data was saved in different .txt files, that is why we load it from parts

outside_data = cell(1,102);
for i=1:7
    data_index = i;
    fileName = sprintf('%stx1_%d_z2.txt', filePath_outside, i);
    % Cell one in the set represents transmitter 1 (outside) transmitting
    % and all inside transceivers receiving, that is why we have a total of
    % 105 rows in each cell (105 inside transceivers), and the outside_data
    % has 102 total cells (102 outside transmitters)
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx2_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:12
    data_index = data_index + 1;
    fileName = sprintf('%stx3_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx4_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx5_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:21
    data_index = data_index + 1;
    fileName = sprintf('%stx6_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:8
    data_index = data_index + 1;
    fileName = sprintf('%stx7_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:18
    data_index = data_index + 1;
    fileName = sprintf('%stx8_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx9_%d_z2.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

elseif flag == 3

% Load inside data
inside_data = cell(1,78);
for i=1:78
    fileName = sprintf('%sz3-p%d-rx3.txt', filePath_inside, i);
    inside_data{i} = load(fileName);
    % Cell one in the set represents transceiver 1
    % transmitting and all other transceivers receiving
    % that is why we have a total of 105 cells (105 inside transceivers)
    % and each cell has 105 rows
end

% Load outside data
% Data was saved in different .txt files, that is why we load it from parts

outside_data = cell(1,102);
for i=1:7
    data_index = i;
    fileName = sprintf('%stx1_%d_z3.txt', filePath_outside, i);
    % Cell one in the set represents transmitter 1 (outside) transmitting
    % and all inside transceivers receiving, that is why we have a total of
    % 105 rows in each cell (105 inside transceivers), and the outside_data
    % has 102 total cells (102 outside transmitters)
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx2_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:12
    data_index = data_index + 1;
    fileName = sprintf('%stx3_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx4_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:7
    data_index = data_index + 1;
    fileName = sprintf('%stx5_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:21
    data_index = data_index + 1;
    fileName = sprintf('%stx6_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:8
    data_index = data_index + 1;
    fileName = sprintf('%stx7_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:18
    data_index = data_index + 1;
    fileName = sprintf('%stx8_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

for i=1:11
    data_index = data_index + 1;
    fileName = sprintf('%stx9_%d_z3.txt', filePath_outside, i);
    outside_data{data_index} = load(fileName);
end

end

end