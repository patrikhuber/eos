%% Demo for running the eos fitting from Matlab
%
%% Set up some required paths to files:
model_file = '../share/sfm_shape_3448.bin';
blendshapes_file = '../share/expression_blendshapes_3448.bin';

%% Read some ibug landmarks for an image:
landmarks = read_pts_landmarks('D:\Github\data\ibug\helen\testset\30427236_1.pts');

%% Run the fitting, get back the fitted mesh and pose:
[mesh, render_params] = eos.fitting.fit_shape_and_pose(model_file, blendshapes_file, landmarks);

%% Visualise the fitted mesh using your favourite tool, for example...
figure(1);
plot3(mesh.vertices(:,1), mesh.vertices(:,2), mesh.vertices(:,3), '.');
% or...
FV.vertices = mesh.vertices(:,1:3);
FV.faces = mesh.tvi;
figure(2);
patch(FV, 'FaceColor', [1 1 1], 'EdgeColor', 'none', 'FaceLighting', 'phong'); light; axis equal; axis off;

%% Visualise the fitting in 2D, on top of the input image:
% First we project all vertices to 2D:
points_2d = zeros(size(mesh.vertices, 1), 2);
h = 720; w = 1280;
viewport = [0, h, w, -h];
for i=1:size(mesh.vertices, 1)
    tmp = render_params.projection*render_params.modelview*mesh.vertices(i, :)';
    	tmp = tmp .* 0.5 + 0.5;
		tmp(1) = tmp(1) * viewport(3) + viewport(1);
		tmp(2) = tmp(2) * viewport(4) + viewport(2);
        points_2d(i, :) = tmp(1:2);
end
% Load the image, plot the projected mesh points on top of it:
img = imread('D:\Github\data\ibug\helen\testset\30427236_1.jpg');
figure(3);
imshow(img);
hold on;
plot(points_2d(:,1), points_2d(:,2), 'g.');
% We can also plot the landmarks the mesh was fitted to:
plot(landmarks(:,1), landmarks(:,2), 'ro');


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