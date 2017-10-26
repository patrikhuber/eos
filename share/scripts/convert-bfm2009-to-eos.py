import numpy as np
import eos
import scipy.io

# This script converts the Basel Face Model 2009 (BFM2009, [1]) to the eos model format,
# specifically the file PublicMM1/01_MorphableModel.mat from the BFM2009 distribution.
#
# The script does not use or convert the segments of the BFM2009, just the global PCA.
# The BFM2009 also does not come with texture (uv-) coordinates. If you have texture coordinates for the BFM, they can be
# added to the eos.morphablemodel.MorphableModel(...) constructor in the third argument. Note that eos only supports one
# uv-coordinate per vertex.
#
# [1]: A 3D Face Model for Pose and Illumination Invariant Face Recognition,
#      P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter,
#      AVSS 2009.
#      http://faces.cs.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model

# Set this to the path of the PublicMM1/01_MorphableModel.mat file from the BFM2009 distribution:
bfm2009_path = r"./PublicMM1/01_MorphableModel.mat"
bfm2009 = scipy.io.loadmat(bfm2009_path)

# The PCA shape model:
# Note: All the matrices are of type float32, so we're good and don't need to convert anything.
shape_mean = bfm2009['shapeMU']
shape_orthogonal_pca_basis = bfm2009['shapePC']
# Their basis is unit norm: np.linalg.norm(shape_pca_basis[:,0]) == 1.0
# And the basis vectors are orthogonal: np.dot(shape_pca_basis[:,0], shape_pca_basis[:,0]) == 1.0
#                                       np.dot(shape_pca_basis[:,0], shape_pca_basis[:,1]) == 1e-08
shape_pca_standard_deviations = bfm2009['shapeEV'] # These are standard deviations, not eigenvalues!
shape_pca_eigenvalues = np.square(shape_pca_standard_deviations)
triangle_list = bfm2009['tl'] - 1 # Convert from 1-based Matlab indexing to 0-based C++ indexing
# The BFM has front-facing triangles defined the wrong way round (not in accordance with OpenGL) - we swap the indices:
for t in triangle_list:
    t[1], t[2] = t[2], t[1]
shape_model = eos.morphablemodel.PcaModel(shape_mean, shape_orthogonal_pca_basis, shape_pca_eigenvalues, triangle_list.tolist())

# PCA colour model:
color_mean = bfm2009['texMU']
# The BFM2009's colour data is in the range [0, 255], while the SFM is in [0, 1], so we divide by 255:
color_mean /= 255
color_orthogonal_pca_basis = bfm2009['texPC']
color_pca_standard_deviations = bfm2009['texEV'] # Again, these are standard deviations, not eigenvalues
color_pca_standard_deviations /= 255 # Divide the standard deviations by the same amount as the mean
color_pca_eigenvalues = np.square(color_pca_standard_deviations)

color_model = eos.morphablemodel.PcaModel(color_mean, color_orthogonal_pca_basis, color_pca_eigenvalues, triangle_list.tolist())

# Construct and save the BFM2009 model in the eos format:
model = eos.morphablemodel.MorphableModel(shape_model, color_model, []) # uv-coordinates can be added here
eos.morphablemodel.save_model(model, "bfm2009.bin")
print("Converted and saved model as bfm2009.bin.")
