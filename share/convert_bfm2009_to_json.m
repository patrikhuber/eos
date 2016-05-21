% Converts the 2009 Basel Face Model (BFM, [1]) to a json file that can be
% read by the eos cereal importer. The json-to-cereal-binary app can
% subsequently be used to generate a small eos .bin file.
%
% [1]: A 3D Face Model for Pose and Illumination Invariant Face
% Recognition, P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter,
% AVSS 2009.
% http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model
%
% Notes:
%  - The script takes quite a while to run (>= 10 minutes)
%  - Produces quite unoptimised json (and a large file). Check with cereal
%    documentation if that can be improved.
%
% Developer notes:
%  - The BFM data type is single, SFM is double? Does json make a difference?
%  - Sort out (un)normalised basis, which one is stored in the BFM?
%  - Domains:
%    Colour: BFM:  [0, 255], SFM: [0, 1].
%    Shape: BFM: in mm (e.g. 50000), SFM: in cm, e.g. 50.
%  - Texture coordinates (model.texture_coordinates) would be saved in the
%    same way as triangle_list, but the BFM doesn't have any.
%  - BFM Matlab file contains the "unnormalised", orthonormal bases (as do
%    the Surrey .scm files).
%
function [] = convert_bfm2009_to_json(bfm_file, json_out_file)

if (~exist('bfm_file', 'var'))
    bfm_file = 'D:/Github/data/bfm/PublicMM1/01_MorphableModel.mat';
end
if (~exist('json_out_file', 'var'))
    json_out_file = 'bfm.json';
end

bfm = load(bfm_file);

% Leave 'nt' on the default. This is only to produce a small output model
% for testing purposes. It'll result in only part of the mesh.
nt = size(bfm.shapeMU, 1); % num triangles times 3
nb = size(bfm.shapePC, 2);

model.cereal_class_version = 0;
model.shape_model.mean.data = bfm.shapeMU(1:nt);
model.shape_model.normalised_pca_basis.data = normalise_pca_basis(bfm.shapePC(1:nt, 1:nb), bfm.shapeEV(1:nb));
model.shape_model.unnormalised_pca_basis.data = bfm.shapePC(1:nt, 1:nb);
model.shape_model.eigenvalues.data = bfm.shapeEV(1:nb);
model.shape_model.triangle_list = {}; % will be populated below
model.color_model.mean.data = bfm.texMU(1:nt);
model.color_model.normalised_pca_basis.data = normalise_pca_basis(bfm.texPC(1:nt, 1:nb), bfm.texEV(1:nb));
model.color_model.unnormalised_pca_basis.data = bfm.texPC(1:nt, 1:nb);
model.color_model.eigenvalues.data = bfm.texEV(1:nb);
model.color_model.triangle_list = {}; % will be populated below
model.texture_coordinates = {}; % the BFM doesn't have any texcoords

model.shape_model.mean.data = model.shape_model.mean.data / 1000;
model.color_model.mean.data = model.color_model.mean.data / 255;
% Divide the basis? The Eigenvectors?
% For the normalised basis, divide before or after the normalisation?

for i = 1:length(bfm.tl)
    v0 = bfm.tl(i, 1) - 1;
    v1 = bfm.tl(i, 2) - 1;
    v2 = bfm.tl(i, 3) - 1;
    if (v0 >= nt/3 || v1 >= nt/3 || v2 >= nt/3)
        continue;
    end
    t.value0 = v0;
    t.value1 = v1;
    t.value2 = v2;
    model.shape_model.triangle_list{i} = t;
    model.color_model.triangle_list{i} = t;
end

bfm_json = savejson('morphable_model', model, json_out_file);

end

% Taken 1:1 from include/eos/morphablemodel/PcaModel.hpp:
%
% * Takes an unnormalised PCA basis matrix (a matrix consisting
% * of the eigenvectors and normalises it, i.e. multiplies each
% * eigenvector by the square root of its corresponding
% * eigenvalue.
% *
% * @param[in] unnormalised_basis An unnormalised PCA basis matrix.
% * @param[in] eigenvalues A row or column vector of eigenvalues.
% * @return The normalised PCA basis matrix.
function [normalised_basis] = normalise_pca_basis(unnormalised_basis, eigenvalues)
	normalised_basis = zeros(size(unnormalised_basis));
	for i = 1:length(eigenvalues)
		sqrt_of_eigenvalues(i) = sqrt(eigenvalues(i));
    end
	% Normalise the basis: We multiply each eigenvector (i.e. each column) with the square root of its corresponding eigenvalue
	for basis = 1:size(unnormalised_basis, 2)
		normalised_eigenvector = unnormalised_basis(:, basis).*sqrt_of_eigenvalues(basis);
		normalised_basis(:, basis) = normalised_eigenvector;
    end
end
