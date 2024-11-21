function plot_rooms(offset_x, offset_y, zone)

hold on

translate_x = + 0.085; % For translation
translate_y = - 0.085; % For translation

x_1 = 504.8 + translate_x;
y_1 = 717.96 + translate_y;

x_2 = 506.47 + translate_x;
y_2 = 715.5 + translate_y;

x_3 = 507.85 + translate_x;
y_3 = 716.38 + translate_y;

x_4 = 510.16 + translate_x - 0.022;
y_4 = 717.85 + translate_y + 0.022;

x_5 = 508.48 + translate_x;
y_5 = 720.35 + translate_y;

x_6 = 509.8 + translate_x;
y_6 = 713.24 + translate_y;

x_7 = 515.81 + translate_x;
y_7 = 717.07 + translate_y;

x_8 = 514.49 + translate_x;
y_8 = 719.12 + translate_y;

x_9 = 511.45 + translate_x;
y_9 = 723.82 + translate_y;

x_10 = 513.04 + translate_x;
y_10 = 724.86 + translate_y;

x_11 = 513.73 + translate_x;
y_11 = 723.83 + translate_y;

x_12 = 516.39 + translate_x;
y_12 = 725.52 + translate_y;

x_13 = 518.83 + translate_x - 0.10;
y_13 = 721.76 + translate_y + 0.10;

x_14 = 509.59 + translate_x;
y_14 = 721.08 + translate_y;

x_15 = 511.25 + translate_x;
y_15 = 718.53 + translate_y;

x_16 = 510.48 + translate_x;
y_16 = 719.71 + translate_y;

x_17 = 513.03 + translate_x;
y_17 = 721.37 + translate_y;

x_18 = 512.128 + translate_x;
y_18 = 722.77 + translate_y;

x_19 = 512.707 + translate_x;
y_19 = 723.138 + translate_y;

x_20 = 513.785 + translate_x;
y_20 = 720.155 + translate_y;

x_21 = 512.12 + translate_x;
y_21 = 714.73 + translate_y;

x = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21];
y = [y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11, y_12, y_13, y_14, y_15, y_16, y_17, y_18, y_19, y_20, y_21];

x = x - offset_x;
y = y - offset_y;

x12 = [x(1), x(2)];
y12 = [y(1), y(2)];
plot(x12, y12, 'LineWidth', 3,  'Color', 'k');

x24 = [x(2), x(4)];
y24 = [y(2), y(4)];
plot(x24, y24, 'LineWidth', 3,  'Color', 'k');

x45 = [x(4), x(5)];
y45 = [y(4), y(5)];
plot(x45, y45, 'LineWidth', 3,  'Color', 'k');

x141 = [x(14), x(1)];
y141 = [y(14), y(1)];
plot(x141, y141, 'LineWidth', 3,  'Color', 'k');

x36 = [x(3), x(6)];
y36 = [y(3), y(6)];
plot(x36, y36, 'LineWidth', 3,  'Color', 'k');

x67 = [x(6), x(7)];
y67 = [y(6), y(7)];
plot(x67, y67, 'LineWidth', 3,  'Color', 'k');

x79 = [x(7), x(9)];
y79 = [y(7), y(9)];
plot(x79, y79, 'LineWidth', 3,  'Color', 'k');

x1716 = [x(17), x(16)];
y1716 = [y(17), y(16)];
plot(x1716, y1716, 'LineWidth', 3,  'Color', 'k');

x1415 = [x(14), x(15)];
y1415 = [y(14), y(15)];
plot(x1415, y1415, 'LineWidth', 3,  'Color', 'k');

x910 = [x(9), x(10)];
y910 = [y(9), y(10)];
plot(x910, y910, 'LineWidth', 3,  'Color', 'k');

x1011 = [x(10), x(11)];
y1011 = [y(10), y(11)];
plot(x1011, y1011, 'LineWidth', 3,  'Color', 'k');

x1112 = [x(11), x(12)];
y1112 = [y(11), y(12)];
plot(x1112, y1112, 'LineWidth', 3,  'Color', 'k');

x138 = [x(13), x(8)];
y138 = [y(13), y(8)];
plot(x138, y138, 'LineWidth', 3,  'Color', 'k');

x1213 = [x(12), x(13)];
y1213 = [y(12), y(13)];
plot(x1213, y1213, 'LineWidth', 3,  'Color', 'k');

x1819 = [x(18), x(19)];
y1819 = [y(18), y(19)];
plot(x1819, y1819, 'LineWidth', 3,  'Color', 'k');

if zone == 1
    h1 = fill([x(14), x(5), x(4), x(15)], [y(14), y(5), y(4), y(15)], 'r', 'FaceAlpha', 0.3);
    h2 = fill([x(16), x(15), x(20), x(17)], [y(16), y(15), y(20), y(17)], 'g', 'FaceAlpha', 0.3);
    h3 = fill([x(3), x(6), x(21), x(4)], [y(3), y(6), y(21), y(4)], 'b', 'FaceAlpha', 0.3);
    h4 = fill([x(4), x(15), x(20), x(7), x(21)], [y(4), y(15) y(20), y(7), y(21)], 'm', 'FaceAlpha', 0.3);
    legend([h1, h2, h3, h4], {'Subregion 1', 'Subregion 2', 'Subregion 3', 'Subregion 4'});
elseif zone == 2
    h1 = fill([x(1), x(2), x(4), x(5)], [y(1), y(2), y(4), y(5)], 'r', 'FaceAlpha', 0.3);
    legend(h1, {'Subregion 1'});
elseif zone == 3
    h1 = fill([x(9), x(10), x(11), x(18)], [y(9), y(10), y(11), y(18)], 'r', 'FaceAlpha', 0.3);
    h2 = fill([x(18), x(11), x(12), x(13), x(8)], [y(18), y(11), y(12), y(13), y(8)], 'g', 'FaceAlpha', 0.3);
    legend([h1, h2], {'Subregion 1', 'Subregion 2'});
end

axis equal;

% X_coordinates_voronoi = [x(6), x(7), x(17), x(5)];
% Y_coordinates_voronoi = [y(6), y(7), y(17), y(5)];
% voronoi(X_coordinates_voronoi,Y_coordinates_voronoi);

end