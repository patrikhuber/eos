% Converts the 2009 Basel Face Model (BFM, [1]) to a binary file that can be
% directly read byte for byte in C++. The workflow is to feed this
% generated "raw" binary file into the bfm-binary-to-cereal app, which
% reads this binary byte for byte and converts it to a cereal .bin file
% that is then readable by eos.
%
% [1]: A 3D Face Model for Pose and Illumination Invariant Face
% Recognition, P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter,
% AVSS 2009.
% http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model
%
% Developer notes:
%  - The BFM data type is single, SFM is double
%  - The BFM Matlab file contains the "unnormalised", orthonormal basis
%    (as do the Surrey .scm files).
%  - Domains:
%    Colour: BFM: [0, 255], SFM: [0, 1].
%    Shape: BFM: in mm (e.g. 50000), SFM: in cm, e.g. 50.
%    (Note: I think that's wrong, since we have to divide by 1000.)
%  - The BFM doesn't have any texture coordinates.
%
function [] = convert_bfm2009_to_raw_binary(bfm_file, binary_out_file)

if (~exist('bfm_file', 'var'))
    bfm_file = 'D:/Github/data/bfm/PublicMM1/01_MorphableModel.mat';
end
if (~exist('binary_out_file', 'var'))
    binary_out_file = 'bfm.raw';
end

bfm = load(bfm_file);

f = fopen(binary_out_file, 'w');

fwrite(f, size(bfm.shapeMU, 1), 'int32'); % num vertices times 3
fwrite(f, size(bfm.shapePC, 2), 'int32'); % number of basis vectors

% Write the shape mean:
fwrite(f, bfm.shapeMU, 'float');

% Write the orthonormal shape PCA basis matrix:
% All of basis 1 will be written first, then basis 2, etc.
fwrite(f, bfm.shapePC, 'float');

% Write the shape eigenvalues:
fwrite(f, bfm.shapeEV, 'float');

% Write num_triangles and the triangle list:
fwrite(f, size(bfm.tl, 1), 'int32');
fwrite(f, bfm.tl', 'int32');

% Now just exactly the same for the colour (albedo) model:
fwrite(f, size(bfm.texMU, 1), 'int32'); % num vertices times 3
fwrite(f, size(bfm.texPC, 2), 'int32'); % number of basis vectors

% Write the colour mean:
fwrite(f, bfm.texMU, 'float');

% Write the orthonormal colour PCA basis matrix:
% All of basis 1 will be written first, then basis 2, etc.
fwrite(f, bfm.texPC, 'float');

% Write the colour eigenvalues:
fwrite(f, bfm.texEV, 'float');

fclose(f);

end
