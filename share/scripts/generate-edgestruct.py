import numpy as np
import eos
import scipy.io

# This script computes an edge_topology.json file for a given model, which is used in eos's contour fitting.
# The script can be used for any Morphable Model, for example the SFM, BFM2009, BFM2017, and others.

# Set this to the path of the model that you want to generate an edgestruct from:
model_path = "bfm2017-1_bfm_nomouth.bin"

# Step 1:
# Save the triangle list of the model to Matlab (to be read by Matlab in Step 2):
model = eos.morphablemodel.load_model(model_path)
triangle_list = np.array(model.get_shape_model().get_triangle_list()) + 1 # add 1 to make 1-based indices for Matlab
scipy.io.savemat("bfm2017-1_bfm_nomouth_trianglelist.mat", {'triangle_list': triangle_list})

# Step 2:
# Open Matlab and run compute_edgestruct.m on the generated triangle-list .mat file.
# Matlab will save an edgestruct.mat file with face and vertex adjacency information.

# Step 3:
# Load the generated edgestruct.mat from Matlab and save it as an eos EdgeTopology in json format:
edgestruct_path = r"edgestruct.mat"
edge_info = scipy.io.loadmat(edgestruct_path)
Ef = edge_info['Ef']
Ev = edge_info['Ev']
edge_topology = eos.morphablemodel.EdgeTopology(Ef.tolist(), Ev.tolist())
eos.morphablemodel.save_edge_topology(edge_topology, "bfm2017-1_bfm_nomouth_edge_topology.json")

print("Finished generating edge-topology file and saved it as bfm2017-1_bfm_nomouth_edge_topology.json.")
