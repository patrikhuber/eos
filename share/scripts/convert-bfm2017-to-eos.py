import numpy as np
import eos
import h5py

# This script converts the Basel Face Model 2017 (BFM2017, [1]) to the eos model format,
# specifically the files model2017-1_face12_nomouth.h5 and model2017-1_bfm_nomouth.h5 from the BFM2017 download.
#
# The BFM2017 does not come with texture (uv-) coordinates. If you have texture coordinates for the BFM, they can be
# added to the eos.morphablemodel.MorphableModel(...) constructor in the third argument. Note that eos only supports one
# uv-coordinate per vertex.
#
# [1]: Morphable Face Models - An Open Framework,
#      T. Gerig, A. Morel-Forster, C. Blumer, B. Egger, M. Lüthi, S. Schönborn and T. Vetter,
#      arXiv preprint, 2017.
#      http://faces.cs.unibas.ch/bfm/bfm2017.html

# Set this to the path of the model2017-1_bfm_nomouth.h5 or model2017-1_face12_nomouth.h5 file from the BFM2017 download:
bfm2017_file = r"./model2017-1_bfm_nomouth.h5"

with h5py.File(bfm2017_file, 'r') as hf:
    # The PCA shape model:
    shape_mean = np.array(hf['shape/model/mean'])
    shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
    # Their basis is unit norm: np.linalg.norm(shape_pca_basis[:,0]) == ~1.0
    # And the basis vectors are orthogonal: np.dot(shape_pca_basis[:,0], shape_pca_basis[:,0]) == 1.0
    #                                       np.dot(shape_pca_basis[:,0], shape_pca_basis[:,1]) == 1e-10
    shape_pca_variance = np.array(hf['shape/model/pcaVariance']) # the PCA variances are the eigenvectors

    triangle_list = np.array(hf['shape/representer/cells'])

    shape_model = eos.morphablemodel.PcaModel(shape_mean, shape_orthogonal_pca_basis, shape_pca_variance, triangle_list.transpose().tolist())

    # PCA colour model:
    color_mean = np.array(hf['color/model/mean'])
    color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
    color_pca_variance = np.array(hf['color/model/pcaVariance'])

    color_model = eos.morphablemodel.PcaModel(color_mean, color_orthogonal_pca_basis, color_pca_variance, triangle_list.transpose().tolist())

    # PCA expression model:
    # The eos master branch does not support PCA expression models yet - thus we don't convert the expression model for now.
    #expression_mean = np.array(hf['expression/model/mean'])
    #expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
    #expression_pca_variance = np.array(hf['expression/model/pcaVariance'])

    #expression_model = eos.morphablemodel.PcaModel(expression_mean, expression_pca_basis, expression_pca_variance, triangle_list.transpose().tolist())

    # Construct and save an eos model from the BFM data:
    model = eos.morphablemodel.MorphableModel(shape_model, color_model, []) # uv-coordinates can be added here
    eos.morphablemodel.save_model(model, "bfm2017-1_bfm_nomouth.bin")
    print("Converted and saved model as bfm2017-1_bfm_nomouth.bin.")
