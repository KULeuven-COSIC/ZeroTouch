function draw_figure2(data_inside, outside_coordinates, offset_x, offset_y, flag, wrong_position_inside_number, wrong_position_outside_number, features_to_use)


X_coordinates = data_inside(:,2);
Y_coordinates = data_inside(:,3);


figure
set(gca, 'Color', 'white'); % Set the background of the plot to white
set(gcf, 'Color', 'white'); % Set the figure background to white
axis off;

for i=1:length(X_coordinates)
    if ismember(i,features_to_use)
    plot(X_coordinates(i),Y_coordinates(i), 's', 'Color', '#0000FF', 'MarkerSize', 9, 'MarkerFaceColor', '#0000FF');
    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'b');
    hold on
    else
    plot(X_coordinates(i),Y_coordinates(i), '.', 'Color', '#000000', 'MarkerSize', 13);
    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'k');
    hold on   
    end
end

X_coordinates = outside_coordinates(:,1);
Y_coordinates = outside_coordinates(:,2);

for i=1:length(X_coordinates)
    plot(X_coordinates(i),Y_coordinates(i), '.', 'Color', '#000000', 'MarkerSize', 13);
    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'k');
    hold on
end

if flag == 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
X_coordinates = data_inside(:,2);
Y_coordinates = data_inside(:,3);

for i=1:length(X_coordinates)
    if ismember(i,wrong_position_inside_number)
    plot(X_coordinates(i),Y_coordinates(i),'r.', 'MarkerSize', 8);
    plot(X_coordinates(i),Y_coordinates(i),'pentagram', 'Color', '#FF0000', 'MarkerSize', 14, 'MarkerFaceColor', '#FF0000');

    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'r');
    hold on
    else
        if ~ismember(i, features_to_use)
            plot(X_coordinates(i),Y_coordinates(i), '.', 'Color', '#000000', 'MarkerSize', 13);
            % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'k');
        end
        hold on 
    end
end

X_coordinates = outside_coordinates(:,1);
Y_coordinates = outside_coordinates(:,2);

for i=1:length(X_coordinates)
    if ismember(i,wrong_position_outside_number)
          plot(X_coordinates(i),Y_coordinates(i),'r.', 'MarkerSize', 8);
          plot(X_coordinates(i),Y_coordinates(i),'pentagram', 'Color', '#FF0000', 'MarkerSize', 14, 'MarkerFaceColor', '#FF0000');
          % plot(X_coordinates(i),Y_coordinates(i),'r*', 'MarkerSize', 12, 'MarkerFaceColor', '#FF0000');
    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'r');
    hold on
    else
    plot(X_coordinates(i),Y_coordinates(i), '.', 'Color', '#000000', 'MarkerSize', 13);
    % text(X_coordinates(i) + 0.08, Y_coordinates(i), num2str(i), 'FontSize', 10, 'Color', 'k');
    hold on
    end
end

end

plot_rooms(offset_x, offset_y, 0);
axis off;

% if flag == 0
% title("Localisation environment - base")
% else
% title("Localisation environment - mistakes (red dot = mistake)")
% end

hold on


end