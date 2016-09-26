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
for i=1:size(bfm.shapeMU, 1)
    fwrite(f, bfm.shapeMU(i), 'float');
end

% Write the unnormalised shape PCA basis matrix:
% All of basis 1 will be written first, then basis 2, etc.
for basis=1:size(bfm.shapePC, 2)
    for j=1:size(bfm.shapePC, 1) % all data points of the basis
        fwrite(f, bfm.shapePC(j, basis), 'float');
    end
end

% Write the shape eigenvalues:
for i=1:size(bfm.shapeEV, 1)
    fwrite(f, bfm.shapeEV(i), 'float');
end

% Write num_triangles and the triangle list:
fwrite(f, size(bfm.tl, 1), 'int32');
for i=1:size(bfm.tl, 1)
    fwrite(f, bfm.tl(i, 1), 'int32');
    fwrite(f, bfm.tl(i, 2), 'int32');
    fwrite(f, bfm.tl(i, 3), 'int32');
end

% Now just exactly the same for the colour (albedo) model:
fwrite(f, size(bfm.texMU, 1), 'int32'); % num vertices times 3
fwrite(f, size(bfm.texPC, 2), 'int32'); % number of basis vectors

% Write the colour mean:
for i=1:size(bfm.texMU, 1)
    fwrite(f, bfm.texMU(i), 'float');
end

% Write the unnormalised colour PCA basis matrix:
% All of basis 1 will be written first, then basis 2, etc.
for basis=1:size(bfm.texPC, 2)
    for j=1:size(bfm.texPC, 1) % all data points of the basis
        fwrite(f, bfm.texPC(j, basis), 'float');
    end
end

% Write the colour eigenvalues:
for i=1:size(bfm.texEV, 1)
    fwrite(f, bfm.texEV(i), 'float');
end

fclose(f);

end
