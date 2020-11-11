# This script computes an edge_topology.json file for a given model, which is
# used in eos's contour fitting. It can be used for any Morphable Model,
# for example the SFM, BFM2009, BFM2017, and others.
# The general idea of this code has been taken from:
# https://github.com/waps101/3DMM_edges:
# A. Bas, W.A.P. Smith, T. Bolkart and S. Wuhrer,
# "Fitting a 3D Morphable Model to Edges: A Comparison Between Hard and Soft
#  Correspondences", ACCV Workshop 2016.


# Set input and output paths for the model that you want to process:
MODEL_PATH = "sfm_shape_3448.bin"
JSON_PATH = "bfm2017-1_bfm_nomouth_edge_topology.json"


def generate_edge_topology(triangles):
    print("triangles:", len(triangles))
    edges = []
    indices = []

    for idx, (v0, v1, v2) in enumerate(triangles, start=1):
        # add 1 to all indices to make 1-based indices for Matlab
        v0, v1, v2 = sorted([v0+1, v1+1, v2+1])
        for e in [[v0, v1], [v0, v2], [v1, v2]]:
            try:
                i = edges.index(e)
            except ValueError:
                edges.append(e)
                indices.append([0, idx])
            else:
                indices[i] = [indices[i][1], idx]
    print("edges:", len(edges), len(indices))
    return indices, edges


if __name__ == "__main__":
    import eos

    model = eos.morphablemodel.load_model(MODEL_PATH)
    triangles = model.get_shape_model().get_triangle_list()
    indices, edges = generate_edge_topology(triangles)

    # Save it as an eos EdgeTopology in json format:
    edge_topology = eos.morphablemodel.EdgeTopology(indices, edges)
    eos.morphablemodel.save_edge_topology(edge_topology, JSON_PATH)
    print("Finished generating edge topology file and saved it as", JSON_PATH)
