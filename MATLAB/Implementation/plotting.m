close all
clear all
clc

load('..\Measurements\Plotting Data\FNR_7dB.mat');
load('..\Measurements\Plotting Data\FNR_7dB_majority.mat');
load('..\Measurements\Plotting Data\FPR_7dB.mat');
load('..\Measurements\Plotting Data\FPR_7dB_majority.mat');

load('..\Measurements\Plotting Data\FNR_sweep.mat');
load('..\Measurements\Plotting Data\FNR_sweep_majority.mat');
load('..\Measurements\Plotting Data\FPR_sweep.mat');
load('..\Measurements\Plotting Data\FPR_sweep_majority.mat');

load('..\Measurements\Plotting Data\thresholds_7dB.mat');
load('..\Measurements\Plotting Data\thresholds_sweep.mat');

% FPR_7dB and FPR_sweep are pretty much identical (sweeping attacker power does not change it)
% the only difference between the two arrays is that the treshold values
% they have been generated on, so to make the plotting more clean we take
% only one curve

% % figure;
% % plot(thresholds_7dB, FPR_7dB, 'g', 'LineWidth', 2, 'DisplayName', 'False Positives');
% % hold on;
% % plot(thresholds_7dB, FNR_7dB, 'b', 'LineWidth', 2, 'DisplayName', 'False Negatives');
% % legend show;
% % xlabel('Threshold value');
% % ylabel('Error Rate');
% % title('False Positives and False Negatives curves for all samples (7 dB)')
% % hold off;
% % 
% % figure;
% % plot(thresholds_7dB, FPR_7dB_majority, 'g', 'LineWidth', 2, 'DisplayName', 'False Positives');
% % hold on;
% % plot(thresholds_7dB, FNR_7dB_majority, 'b', 'LineWidth', 2, 'DisplayName', 'False Negatives');
% % legend show;
% % xlabel('Threshold value');
% % ylabel('Error Rate');
% % title('False Positives and False Negatives curves for majority voting (7 dB)')
% % hold off;
% % 
% % figure;
% % plot(thresholds_7dB, FPR_7dB, 'g', 'LineWidth', 2, 'DisplayName', 'False Positives');
% % hold on;
% % plot(thresholds, FNR_sweep, 'b', 'LineWidth', 2, 'DisplayName', 'False Negatives');
% % legend show;
% % xlabel('Threshold value');
% % ylabel('Error Rate');
% % title('False Positives and False Negatives curves for all samples (power sweep)')
% % hold off;
% % 
% % figure;
% % plot(thresholds_7dB, FPR_7dB_majority, 'g', 'LineWidth', 2, 'DisplayName', 'False Positives');
% % hold on;
% % plot(thresholds, FNR_sweep_majority, 'b', 'LineWidth', 2, 'DisplayName', 'False Negatives');
% % legend show;
% % xlabel('Threshold value');
% % ylabel('Error Rate');
% % title('False Positives and False Negatives curves for majority voting (power sweep)')
% % hold off;

% figure('Units', 'inches', 'Position', [1, 1, 19.8, 13.2]); % [x, y, width, height]
% plot(thresholds_7dB, FPR_7dB, 'Color', [0 0 0], 'LineWidth', 1.9, 'LineStyle', '-', 'DisplayName', 'False Positives (7 dB and sweep)'); % Black
% hold on;
% plot(thresholds_7dB, FNR_7dB, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.9, 'LineStyle', '--', 'DisplayName', 'False Negatives (7 dB)'); % Gray
% hold on;
% plot(thresholds, FNR_sweep, 'Color', [0 0 1], 'LineWidth', 1.9, 'LineStyle', ":", 'DisplayName', 'False Negatives (sweep)'); % Bluelegend show;
% legend show;
% legend('Location', 'best');
% xlabel('Threshold value');
% ylabel('Error Rate');
% xlim([2.00823, 7.5]); 
% ylim([0, 1]);
% set(gcf, 'Color', 'white');
% set(gca, 'Color', 'white');
% ax = gca; 
% ax.FontSize = 19;
% % Adjust figure properties
% set(gcf, 'PaperUnits', 'inches'); % Use inches for sizing
% set(gcf, 'PaperPosition', [0 0 19.8 13.2]); % [left, bottom, width, height]
% set(gcf, 'PaperSize', [19.8 13.2]); % Match paper size to figure size
% print(gcf, 'errors_all.pdf', '-dpdf');
% hold off;
% 
figure('Units', 'inches', 'Position', [1, 1, 19.8, 13.2]); % [x, y, width, height]
plot(thresholds_7dB, FPR_7dB_majority, 'Color', [0 0 0], 'LineWidth', 1.9, 'LineStyle', '-', 'DisplayName', 'False Positives (7 dB and sweep)'); % Black
hold on;
plot(thresholds_7dB, FNR_7dB_majority, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.9, 'LineStyle', '--', 'DisplayName', 'False Negatives (7 dB)'); % Gray
hold on;
plot(thresholds, FNR_sweep_majority, 'Color', [0 0 1], 'LineWidth', 1.9, 'LineStyle', ":", 'DisplayName', 'False Negatives (sweep)'); % Blue
legend show;
legend('Location', 'best');
xlabel('Threshold value');
ylabel('Error Rate');
xlim([3, 7]); 
ylim([0, 1]);
set(gcf, 'Color', 'white');
set(gca, 'Color', 'white');
ax = gca;
ax.FontSize = 30;
% Adjust figure properties
set(gcf, 'PaperUnits', 'inches'); % Use inches for sizing
set(gcf, 'PaperPosition', [0 0 19.8 13.2]); % [left, bottom, width, height]
set(gcf, 'PaperSize', [19.8 13.2]); % Match paper size to figure size
print(gcf, 'errors_majority.pdf', '-dpdf');
hold off;

random_p_means_room_1 = [78.81 80.77 84.93 86.97 86.83 91.54 92.81];
random_p_deviation_room_1 = [3.81 4.69 7.56 6.98 6.52 4.99 6.47];

random_f_means_room_1 = [77.67 79.36 83.36 84.67 84.56 89.28 91.24];
random_f_deviation_room_1 = [5.08 4.96 8.72 7.47 8.19 6.35 7.42];

random_p_means_room_2 = [83.51 95.26 97.19 96.84 97.01 97.89 100];
random_p_deviation_room_2 = [11.28 7.26 7.08 6.50 7.02 6.65 0];

random_f_means_room_2 = [92.50 97.02 97.99 98.46 98.66 98.97 100];
random_f_deviation_room_2 = [5.14 3.91 3.89 3.18 3.31 3.25 0];

random_p_means_room_3 = [84.09 84.82 86.00 85.45 87.18 89.09 89.18];
random_p_deviation_room_3 = [4.31 4.20 5.06 5.07 3.28 0.42 0.79];

random_f_means_room_3 = [86.82 87.33 87.56 87.04 88.45 89.27 89.16];
random_f_deviation_room_3 = [4.37 4.19 4.51 4.38 2.62 0.66 0.86];

random_p_means_all = [82.14 86.95 89.37 89.75 90.34 92.84 93.99];
random_f_means_all = [85.66 87.90 89.64 90.06 90.56 92.51 93.46];

positions = 10:5:40;

% figure
% hold on
% boxchart(positions, random_p_means_room_1, 'BoxFaceColor', 'blue');
% errorbar(positions, random_p_means_room_1, random_p_deviation_room_1, 'k.', 'LineWidth', 1.5);
% xlabel('Number od anchor nodes');
% ylabel('Average accuracy');
% hold off


%% Plots (multiple)

% figure
% hold on
% xlabel('Number od anchor nodes');
% ylabel('Average accuracy');
% 
% upper_bound = min(100, random_p_means_room_1 + random_p_deviation_room_1);
% lower_bound = max(0, random_p_means_room_1 - random_p_deviation_room_1);
% h1 = plot(positions, random_p_means_room_1, 'b', 'LineWidth', 1.5);
% plot(positions, upper_bound, 'b--', 'LineWidth', 1);
% plot(positions, lower_bound, 'b--', 'LineWidth', 1);
% fill([positions, fliplr(positions)], [upper_bound, fliplr(lower_bound)], 'b', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% legend(h1, 'Room 1, Random Placement, Proximity', 'Location', 'Best');
% 
% upper_bound = min(100, random_f_means_room_1 + random_f_deviation_room_1);
% lower_bound = max(0, random_f_means_room_1 - random_f_deviation_room_1);
% h2 = plot(positions, random_f_means_room_1, 'g', 'LineWidth', 1.5);
% plot(positions, upper_bound, 'g--', 'LineWidth', 1);
% plot(positions, lower_bound, 'g--', 'LineWidth', 1);
% fill([positions, fliplr(positions)], [upper_bound, fliplr(lower_bound)], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% legend(h2, 'Room 1, Random Placement, Full', 'Location', 'Best');
% 
% upper_bound = min(100, random_p_means_room_2 + random_p_deviation_room_2);
% lower_bound = max(0, random_p_means_room_2 - random_p_deviation_room_2);
% h3 = plot(positions, random_p_means_room_2, 'm', 'LineWidth', 1.5);
% plot(positions, upper_bound, 'm--', 'LineWidth', 1);
% plot(positions, lower_bound, 'm--', 'LineWidth', 1);
% fill([positions, fliplr(positions)], [upper_bound, fliplr(lower_bound)], 'm', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% legend(h3, 'Room 2, Random Placement, Proximity', 'Location', 'Best');
% 
% upper_bound = min(100, random_f_means_room_2 + random_f_deviation_room_2);
% lower_bound = max(0, random_f_means_room_2 - random_f_deviation_room_2);
% h4 = plot(positions, random_f_means_room_2, 'r', 'LineWidth', 1.5);
% plot(positions, upper_bound, 'r--', 'LineWidth', 1);
% plot(positions, lower_bound, 'r--', 'LineWidth', 1);
% fill([positions, fliplr(positions)], [upper_bound, fliplr(lower_bound)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% legend([h1 h2 h3 h4], {'Room 1, Random Placement, Proximity', 'Room 1, Random Placement, Full', 'Room 2, Random Placement, Proximity', 'Room 2, Random Placement, Full'}, 'Location', 'Best');
% 
% hold off
