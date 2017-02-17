function [ texturemap ] = extract_texture(mesh, rendering_params, image, ...
    compute_view_angle, isomap_resolution)
% EXTRACT_TEXTURE  Extracts the texture from an image and returns the texture map.
%   [ texturemap ] = EXTRACT_TEXTURE(mesh, rendering_params, image, ...
%     compute_view_angle, isomap_resolution)
%
%   Extracts the texture of the face from the given image and stores it as
%   isomap (a rectangular texture map).
%
%   Default values for the parameters: compute_view_angle = false,
%   isomap_resolution = 512.
%
%   Please see the C++ documentation for the full description:
%   http://patrikhuber.github.io/eos/doc/ (TODO: Update to v0.10.1!)

% We'll use default values to the following arguments, if they're not
% provided:
if (~exist('compute_view_angle', 'var')), compute_view_angle = false; end
if (~exist('isomap_resolution', 'var')), isomap_resolution = 512; end

[ texturemap ] = render('extract_texture', mesh, rendering_params, image, compute_view_angle, isomap_resolution);

end
