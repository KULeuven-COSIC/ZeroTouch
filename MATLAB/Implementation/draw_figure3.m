function draw_figure3(data_inside, outside_coordinates, offset_x, offset_y, flag, wrong_position_number, points_for_training, points_for_testing, receivers_to_use)


X_coordinates_inside = data_inside(:,2);
Y_coordinates_inside = data_inside(:,3);

X_coordinates_outside = outside_coordinates(:,1);
Y_coordinates_outside = outside_coordinates(:,2);

X_coordinates = [X_coordinates_inside
                 X_coordinates_outside];

Y_coordinates = [Y_coordinates_inside
                 Y_coordinates_outside];

if flag == 0

figure
plot_rooms(offset_x, offset_y, 0);
for i=1:length(X_coordinates_inside)
    if ismember(i,receivers_to_use)
    plot(X_coordinates_inside(i),Y_coordinates_inside(i),'b.', 'MarkerSize', 14);
    text(X_coordinates_inside(i) + 0.08, Y_coordinates_inside(i), num2str(i), 'FontSize', 10, 'Color', 'b');
    else
    plot(X_coordinates_inside(i),Y_coordinates_inside(i),'k.', 'MarkerSize', 14);
    text(X_coordinates_inside(i) + 0.08, Y_coordinates_inside(i), num2str(i), 'FontSize', 10, 'Color', 'k'); 
    end
    hold on
end

title("Localisation environment - receivers used (blue = used)");
hold off

figure
plot_rooms(offset_x, offset_y, 0);
for i=1:length(X_coordinates)
    j = i;
    if j > 105
       j = j - 105;
    end
    if ismember(i,points_for_training)
    plot(X_coordinates(i),Y_coordinates(i),'b.', 'MarkerSize', 14);
    text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(j), 'FontSize', 10, 'Color', 'b');
    else
    plot(X_coordinates(i),Y_coordinates(i),'k.', 'MarkerSize', 14);
    text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(j), 'FontSize', 10, 'Color', 'k'); 
    end
    hold on
end
title("Localisation environment - tranmsitters used for training (blue = used)");
hold off

else

figure
plot_rooms(offset_x, offset_y, 0);
for i=1:length(X_coordinates)
    j = i;
    if j > 105
       j = j - 105;
    end
    if ismember(i,points_for_testing) && ~ismember(i,wrong_position_number)
    plot(X_coordinates(i),Y_coordinates(i),'b.', 'MarkerSize', 14);
    text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(j), 'FontSize', 10, 'Color', 'b');
    elseif ismember(i,wrong_position_number)
    plot(X_coordinates(i),Y_coordinates(i),'r.', 'MarkerSize', 14);
    text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(j), 'FontSize', 10, 'Color', 'r'); 
    else
    plot(X_coordinates(i),Y_coordinates(i),'k.', 'MarkerSize', 14);
    text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(j), 'FontSize', 10, 'Color', 'k'); 
    hold on
    end

end

title("Localisation environment - tranmsitters used for testing (blue = used, red = error)");
hold off
end
