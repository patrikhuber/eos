%% Demo for running the eos fitting from Matlab
%
%% Set up some required paths to files:
model_file = '../share/sfm_shape_3448.bin';
blendshapes_file = '../share/expression_blendshapes_3448.bin';
landmark_mappings = '../share/ibug_to_sfm.txt';

%% Load an image and its landmarks in ibug format:
image = imread('../bin/data/image_0010.png');
landmarks = read_pts_landmarks('../bin/data/image_0010.pts');
image_width = size(image, 2); image_height = size(image, 1);

%% Run the fitting, get back the fitted mesh and pose:
[mesh, render_params] = eos.fitting.fit_shape_and_pose(model_file, blendshapes_file, landmarks, landmark_mappings, image_width, image_height);
% Note: The function actually has a few more arguments to files it
% needs. If you're not running it from within eos/matlab/, you need to
% provide them. See its documentation and .m file.

%% Visualise the fitted mesh using your favourite plot, for example...
figure(1);
plot3(mesh.vertices(:, 1), mesh.vertices(:, 2), mesh.vertices(:, 3), '.');
% or...
FV.vertices = mesh.vertices(:, 1:3);
FV.faces = mesh.tvi;
figure(2);
patch(FV, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong'); light; axis equal; axis off;

%% Visualise the fitting in 2D, on top of the input image:
% Project all vertices to 2D:
points_2d = mesh.vertices * (render_params.viewport*render_params.projection*render_params.modelview)';
% Display the image and plot the projected mesh points on top of it:
figure(3);
imshow(image);
hold on;
plot(points_2d(:, 1), points_2d(:, 2), 'g.');
% We can also plot the landmarks the mesh was fitted to:
plot(landmarks(:, 1), landmarks(:, 2), 'ro');


%% Just a helper function to read ibug .pts landmarks from a file:
function [ landmarks ] = read_pts_landmarks(filename)

file = fopen(filename, 'r');
file_content = textscan(file, '%s');

landmarks = zeros(68, 2);

row_idx = 1;
for i=6:2:141
    landmarks(row_idx, 1) = str2double(file_content{1}{i});
    landmarks(row_idx, 2) = str2double(file_content{1}{i + 1}); 
    row_idx = row_idx + 1;
end

end
