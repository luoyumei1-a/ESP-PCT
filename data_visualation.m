clear;

dir_path="matData\0\a"

% Get a list of all subdirectories within the main directory
subdirs = dir(dir_path);
subdirs = subdirs([subdirs.isdir]);  % Filter out non-directories
subdirs = subdirs(3:end);  % Skip '.' and '..' directories
all=0;
avg=0;

meiluoyu=0;
if meiluoyu==1
    suffix = ".mat_0.mat";
    suffix_ply = ".mat.ply";
else
    suffix = ".mat";
end

saveAsVideo = 1;
figFrameCount = 0;
videofile = "test.avi";
xyzivb_ti_list = {};
xyz_key_list = {};
xyzb_kinect_list = {};
xyz_bounding_box_list = {};

% read the data and find the activity area
x_max = -Inf; y_max = -Inf; z_max = -Inf;
x_min = Inf; y_min = Inf; z_min = Inf;
all_mat_list = {};
hand_points_history = {}; % 存储过去10帧的手部点云数据

% Loop through subdirectories
for subdir_index = 1:numel(subdirs)
    subdir = subdirs(subdir_index).name;
    data_dir = fullfile(dir_path, subdir);
    
    % List MAT files in the current subdirectory
    mat_list = dir(data_dir);    
    % Extract the names of the MAT files
    file_names = {mat_list.name};    
    % Identify the valid MAT files (excluding '.' and '..')
    valid_indices = ~ismember(file_names, {'.', '..'});    
    % Extract the valid file names
    valid_file_names = file_names(valid_indices);    
    % Extract the numeric parts of the file names (xxx.mat)
    file_numbers = cellfun(@(x) sscanf(x, 'pc_ti_kinect_key_%d.mat'), valid_file_names);    
    % Sort the file numbers
    [sorted_numbers, sorted_indices] = sort(file_numbers);    
    % Reorder the MAT list based on sorted indices
    mat_list = mat_list(valid_indices);
    mat_list = mat_list(sorted_indices);

    % Initialize variables for this subdirectory
    upper_point_num = 0;
    lower_point_num = 0;
    
    % Loop through MAT files in the current subdirectory
    for mat_index = 1:numel(mat_list)
        mat_filename = mat_list(mat_index).name;
        if mat_filename=="." || mat_filename==".."
            continue;
        end
        all_mat_list{end+1} = mat_filename;
        if endsWith(mat_filename, '.mat')
            mat_dir = fullfile(data_dir, mat_filename);
            data = load(mat_dir);
    
            % xyzivb_ti = data.radar_point_cloud;
            xyzivb_ti = data.pc_xyziv_ti;
            if size(xyzivb_ti)==0
                continue;
            end
            n_ti=length(xyzivb_ti);
            all=all+n_ti;
            
            xyz_key = data.pc_xyz_key;     
            xyzb_kinect = data.pc_xyzb_kinect;

            xyz_key(:,3) = xyz_key(:,3)-0.1;
            xyzivb_ti(:,1) = xyzivb_ti(:,1)-0.2;
            xyzivb_ti(:,2) = xyzivb_ti(:,2)+0.3;
            xyzivb_ti(:,3) = xyzivb_ti(:,3)-0.9;
                        
            upper_point_num=upper_point_num+numel(find(xyzivb_ti(:,3)>=xyz_key(1,3)+0.12));
            lower_point_num=lower_point_num+numel(find(xyzivb_ti(:,3)<xyz_key(1,3)+0.12));
            
            %多个帧中xyz最小值，最大值
            x_min = min([x_min, min(xyzivb_ti(:,1)), min(xyz_key(:,1))]);
            y_min = min([y_min, min(xyzivb_ti(:,2)), min(xyz_key(:,2))]);
            z_min = min([z_min, min(xyzivb_ti(:,3)), min(xyz_key(:,3))]);
            
            x_max = max([x_max, max(xyzivb_ti(:,1)), max(xyz_key(:,1))]);
            y_max = max([y_max, max(xyzivb_ti(:,2)), max(xyz_key(:,2))]);
            z_max = max([z_max, max(xyzivb_ti(:,3)), max(xyz_key(:,3))]);

            hand_indices = isHandPoint(xyzivb_ti);
            hand_points = xyzivb_ti(hand_indices, :);

            if size(hand_points, 1) < 6 && mat_index > 2
                avg_change = mean(diff(cell2mat(hand_points_history)), 1);
                hand_points = [hand_points; hand_points_history{end} + 3*avg_change];
                xyzivb_ti = [xyzivb_ti; hand_points];
            end

            hand_indices = isHandPoint(xyzivb_ti);
            hand_points = xyzivb_ti(hand_indices, :);
            [xmin, xmax, ymin, ymax, zmin, zmax] = getBoundingBox(hand_points);
            [X,Y,Z] = meshgrid([xmin xmax], [ymin ymax], [zmin zmax]);
            cube = [X(:) Y(:) Z(:)];

            if size(hand_points_history,1) < 5
                hand_points_history = [hand_points_history; hand_points];
            else
                hand_points_history = hand_points_history(2:end,:);
                hand_points_history = [hand_points_history; hand_points];
            end
            
            xyz_bounding_box_list = [xyz_bounding_box_list, cube];
            xyzivb_ti_list = [xyzivb_ti_list, xyzivb_ti];
            xyz_key_list = [xyz_key_list, xyz_key];
            xyzb_kinect_list= [xyzb_kinect_list, xyzb_kinect];
        end
    end
end

% visualize
fig = figure();
set(gcf,'Position',[100 100 2500 400])
for index = 1:length(all_mat_list)
    subplot(1,3,1);
    xyzivb_ti = xyzivb_ti_list{index}; %radar point cloud
    xyz_key = xyz_key_list{index}; %kinect key points
    xyzb_kinect = xyzb_kinect_list{index}; %kinect key points
    bounding_box = xyz_bounding_box_list{index}; %bounding box for current frame

    X_r = xyzivb_ti(:,1);
    Y_r = xyzivb_ti(:,2);
    Z_r = xyzivb_ti(:,3);
    plot3(X_r, Y_r, Z_r, "ro", 'MarkerFaceColor','r')
    xlabel('x(t)')
    ylabel('y(t)')
    zlabel('z(t)')
    % xlim([x_min x_max])
    xlim([-0.4 0.3])
    % ylim([y_min y_max])
    ylim([1.1 1.9])
    zlim([-0.8 0.5]) % Adjust the limits of the z-axis to the range you're interested in.
    % zlim([z_min z_max])
    hold on;
    grid on;
    scatter3(xyzb_kinect(:,1),xyzb_kinect(:,2), xyzb_kinect(:,3),'.','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerEdgeAlpha',0.6);
    hold on;
    grid on;
    drawKinectPose3D(xyz_key, fig)
    % Draw bounding box for first subplot
    drawBoundingBox3D(bounding_box, 'g');
    hold off;

    subplot(1,3,2);
    scatter(xyzb_kinect(:,1), xyzb_kinect(:,2),'.','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerEdgeAlpha',0.6);
    hold on;
    plot(X_r, Y_r, "ro", 'MarkerFaceColor','r')
    xlabel('x(t)')
    ylabel('y(t)')
    % xlim([x_min x_max])
    xlim([-0.4 0.3])
    % ylim([y_min y_max])
    ylim([1.1 1.9])
    hold on;
    drawKinectPoseXY(xyz_key, fig);
    % Draw bounding box for second subplot
    drawBoundingBox2DXY(bounding_box, 'g');
    grid on;
    legend("Kinect Mesh", "Radar", "Joints", 'Location','southwest')

    subplot(1,3,3);
    i = index - 1; 
    if meiluoyu==1
        ptCloud = pcread(dir + prefix + i + suffix_ply);
        kinect_raw_mesh = ptCloud.Location;
        scatter(kinect_raw_mesh(:,2), kinect_raw_mesh(:,3),'.','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerEdgeAlpha',0.6);
    else
        scatter(xyzb_kinect(:,2), xyzb_kinect(:,3),'.','MarkerEdgeColor',[0.8 0.8 0.8],'MarkerEdgeAlpha',0.6);
    end 
    hold on;
    plot(Y_r, Z_r, "ro", 'MarkerFaceColor','r')
    xlabel('y(t)')
    ylabel('z(t)')
    % xlim([y_min y_max])
    xlim([1.1 1.9])
    ylim([-0.8 0.5]) % Adjust the limits of the z-axis to the range you're interested in.
    % ylim([z_min z_max])
    hold on;
    drawKinectPoseYZ(xyz_key, fig);
    % Draw bounding box for second subplot
    drawBoundingBox2DYZ(bounding_box, 'g');
    grid on;
    legend("Kinect Mesh", "Radar", "Joints", 'Location','southwest')

    % record
    figFrameCount = figFrameCount + 1;
    if saveAsVideo
      F(figFrameCount) = getframe(gcf); 
    end
    pause(0.1)
end

function [] = drawKinectPose3D(xyz_key, fig)
    figure(fig)
    parent_joints = 1+ [0,0,1,2,2,4,5,6,7,8,7,2,11,12,13,14,15,14,0,18,19,20,0,22,23,24,3,26,26,26,26,26];

    for idx = 1:size(xyz_key,1)

        joint = [xyz_key(idx,1),xyz_key(idx,2),xyz_key(idx,3)];
        parent =  [xyz_key(parent_joints(idx),1),xyz_key(parent_joints(idx),2),xyz_key(parent_joints(idx),3)];%存储每个关键点对应的父节点
        plot3([joint(1),parent(1)], ...
            [joint(2),parent(2)], ...
            [joint(3),parent(3)], "k-",  "Linewidth", 2);
        hold on;
    end
    hold off
end

function [] = drawKinectPoseXY(xyz_key, fig)
    figure(fig)
    parent_joints = 1+ [0,0,1,2,2,4,5,6,7,8,7,2,11,12,13,14,15,14,0,18,19,20,0,22,23,24,3,26,26,26,26,26];
    for idx = 1:size(xyz_key,1)
        joint = [xyz_key(idx,1),xyz_key(idx,2),xyz_key(idx,3)];
        parent =  [xyz_key(parent_joints(idx),1),xyz_key(parent_joints(idx),2),xyz_key(parent_joints(idx),3)];
        plot([joint(1),parent(1)], [joint(2),parent(2)],"k-", "Linewidth", 2);
        hold on;
    end
    hold off
end

function [] = drawKinectPoseYZ(xyz_key, fig)
    figure(fig)
    parent_joints = 1+ [0,0,1,2,2,4,5,6,7,8,7,2,11,12,13,14,15,14,0,18,19,20,0,22,23,24,3,26,26,26,26,26];
    for idx = 1:size(xyz_key,1)
        joint = [xyz_key(idx,1),xyz_key(idx,2),xyz_key(idx,3)];
        parent =  [xyz_key(parent_joints(idx),1),xyz_key(parent_joints(idx),2),xyz_key(parent_joints(idx),3)];
        plot([joint(2),parent(2)], [joint(3),parent(3)],"k-", "Linewidth", 2);
        hold on;
    end
    hold off
end

function hand_indices = isHandPoint(xyz)
    xmin = -0.2; xmax = 0.2;
    ymin = 1.1; ymax = 1.5;
    zmin = -0.2; zmax = 0.3;

    hand_indices = xyz(:,1) >= xmin & xyz(:,1) <= xmax & ...
                   xyz(:,2) >= ymin & xyz(:,2) <= ymax & ...
                   xyz(:,3) >= zmin & xyz(:,3) <= zmax;
end

function [xmin, xmax, ymin, ymax, zmin, zmax] = getBoundingBox(points)
% Static variables to save the bounding box of the previous frame
persistent last_xmin last_xmax last_ymin last_ymax last_zmin last_zmax
    % Initialize the static variables in the first call
    if isempty(last_xmin)
        % Initialize with some default values
        last_xmin = -Inf;
        last_xmax = Inf;
        last_ymin = -Inf;
        last_ymax = Inf;
        last_zmin = -Inf;
        last_zmax = Inf;
    end
    % Extract the x, y and z coordinates
    x = points(:, 1);
    y = points(:, 2);
    z = points(:, 3);
    
    % Compute the min and max for each dimension
    xmin = min(x);
    xmax = max(x);
    ymin = min(y);
    ymax = max(y);
    zmin = min(z);
    zmax = max(z);

    % If the number of points is less than 4, return the bounding box of the previous frame
    if size(points, 1) < 3
        xmin = last_xmin;
        xmax = last_xmax;
        ymin = last_ymin;
        ymax = last_ymax;
        zmin = last_zmin;
        zmax = last_zmax;
        return;
    end

    % Save the bounding box of the current frame
    last_xmin = xmin;
    last_xmax = xmax;
    last_ymin = ymin;
    last_ymax = ymax;
    last_zmin = zmin;
    last_zmax = zmax;
end

function drawBoundingBox3D(bounding_box, color)
    % Define the faces of the bounding box
    faces = [1 2 4 3; 5 6 8 7; 1 5 7 3; 2 6 8 4; 1 2 6 5; 3 4 8 7];
    
    % Draw the bounding box
    patch('Vertices', bounding_box, 'Faces', faces, 'FaceColor', 'none', 'EdgeColor', color, 'LineWidth', 2);
end

function drawBoundingBox2DXY(bounding_box, color)
    x_min = min(bounding_box(:, 1));
    x_max = max(bounding_box(:, 1));
    y_min = min(bounding_box(:, 2));
    y_max = max(bounding_box(:, 2));

    small_bounding_box = [x_min, y_min; ...
                      x_max, y_min; ...
                      x_max, y_max; ...
                      x_min, y_max];

    % Define the faces of the bounding box
    faces = [1 2 3 4];

    % Draw the bounding box
    patch('Vertices', small_bounding_box, 'Faces', faces, 'FaceColor', 'none', 'EdgeColor', color, 'LineWidth', 2);
end

function drawBoundingBox2DYZ(bounding_box, color)
    y_min = min(bounding_box(:, 2));
    y_max = max(bounding_box(:, 2));
    z_min = min(bounding_box(:, 3));
    z_max = max(bounding_box(:, 3));

    small_bounding_box = [y_min, z_min; ...
                      y_max, z_min; ...
                      y_max, z_max; ...
                      y_min, z_max];

    % Define the faces of the bounding box
    faces = [1 2 3 4];

    % Draw the bounding box
    patch('Vertices', small_bounding_box, 'Faces', faces, 'FaceColor', 'none', 'EdgeColor', color, 'LineWidth', 2);
end