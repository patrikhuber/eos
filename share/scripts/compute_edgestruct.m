%% The code in this file is largely copied and modified from
% https://github.com/waps101/3DMM_edges:
% A. Bas, W.A.P. Smith, T. Bolkart and S. Wuhrer, "Fitting a 3D Morphable
% Model to Edges: A Comparison Between Hard and Soft Correspondences",
% ACCV Workshop 2016.
% The code is licensed under the Apache-2.0 license.

%% Read the instructions in share/generate-edgestruct.py for how to use this script.

function [] = compute_edgestruct(trianglelist_file)
    load(trianglelist_file); % loads 'triangle_list' from the file
    num_vertices = max(max(triangle_list)); % we assume that the largest triangle
                                            % index that we're going to find is the
                                            % number of vertices of the model.
    % Get the edge list:
    TR = triangulation(double(triangle_list), ones(num_vertices, 1), ones(num_vertices, 1), ones(num_vertices, 1));
    Ev = TR.edges; % This should be a list of all the edges.
    clear TR;
    Ef = meshFaceEdges(triangle_list, Ev);
    save('edgestruct.mat', 'Ef', 'Ev'); % Load this file back into generate-edgestruct.py.
end

% This function is copied from:
% https://github.com/waps101/3DMM_edges/blob/master/utils/meshFaceEdges.m,
% on 3 Oct 2016.
function Ef = meshFaceEdges(faces, edges)
%MESHFACEEDGES Compute faces adjacent to each edge in the mesh
%   faces - nverts by 3 matrix of mesh faces
%   edges - nedges by 2 matrix containing vertices adjacent to each edge
%
% This function is slow! But it only needs to be run once for a morphable
% model and the edge-face list can then be saved

nedges = size(edges, 1);

faces = sort(faces, 2);
edges = sort(edges, 2);

disp('      ');
for i=1:nedges
    idx = find(((faces(:,1)==edges(i,1)) & ( (faces(:,2)==edges(i,2)) | (faces(:,3)==edges(i,2)) )) | ((faces(:,2)==edges(i,1)) & (faces(:,3)==edges(i,2))));
    if length(idx)==1
        idx = [0 idx];
    end
    Ef(i,:)=[idx(1) idx(2)];
    fprintf('\b\b\b\b\b\b%05.2f%%', i/nedges*100);
end

end
