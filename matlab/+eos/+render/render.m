function [ framebuffer, depthbuffer ] = render(mesh, modelview_matrix, ...
    projection_matrix, image_width, image_height, texture)
% RENDER  Renders the given mesh in the given pose.
%   [ framebuffer, depthbuffer ] = RENDER(mesh, modelview_matrix, ...
%     projection_matrix, image_width, image_height, texture)
%
%   Renders the mesh with given model-view and projection matrix, and given
%   texture, and returns the rendered framebuffer as well as the depthbuffer.
%
%   Please see the C++ documentation for the full description:
%   http://patrikhuber.github.io/eos/doc/ (TODO: Update to v0.10.1!)

[ framebuffer, depthbuffer ] = render('render', mesh, modelview_matrix, ...
    projection_matrix, image_width, image_height, texture);

end
