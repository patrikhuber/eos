import eos
import numpy as np
from scipy.io import loadmat

# This script loads the Liverpool-York Head Model (LYHM, [1]) from one of their Matlab .mat files into the eos model
# format, and returns an eos.morphablemodel.MorphableModel.
#
# Note: The LYHM does not come with texture (uv-) coordinates. If you have texture coordinates for the model, they can
# be added to the eos.morphablemodel.MorphableModel(...) constructor as a parameter.
#
# [1]: Statistical Modeling of Craniofacial Shape and Texture,
#      H. Dai, N. E. Pears, W. Smith and C. Duncan,
#      International Journal of Computer Vision (2019).
#      https://www-users.cs.york.ac.uk/~nep/research/LYHM/


def load_lyhm(matlab_model_path):
    lyhm = loadmat(matlab_model_path)
    triangle_list = lyhm['tri']['faces'][0][0] - 1  # Convert from 1-based Matlab indexing to 0-based C++ indexing
    # The LYHM has front-facing triangles defined the wrong way round (not in accordance with OpenGL) - we swap the indices:
    for t in triangle_list:
        t[1], t[2] = t[2], t[1]

    # The LYHM .mat files contain the orthonormal basis vectors, so we don't need to convert anything:
    shape_mean = lyhm['shp']['mu'][0][0][0]
    shape_orthonormal_pca_basis = lyhm['shp']['eigVec'][0][0]
    shape_pca_eigenvalues = lyhm['shp']['eigVal'][0][0]

    # The color values are in [0, 1]
    color_mean = lyhm['tex']['mu'][0][0][0]
    color_orthonormal_pca_basis = lyhm['tex']['eigVec'][0][0]
    color_pca_eigenvalues = lyhm['tex']['eigVal'][0][0]

    # Construct and return the LYHM as eos MorphableModel:
    shape_model = eos.morphablemodel.PcaModel(shape_mean, shape_orthonormal_pca_basis, shape_pca_eigenvalues, triangle_list)
    color_model = eos.morphablemodel.PcaModel(color_mean, color_orthonormal_pca_basis, color_pca_eigenvalues, triangle_list)
    model = eos.morphablemodel.MorphableModel(shape_model, color_model)

    return model
