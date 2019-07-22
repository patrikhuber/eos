# This script computes an edge_topology.json file for a given model, which is used in eos's contour fitting.
# The script can be used for any Morphable Model, for example the SFM, BFM2009, BFM2017, and others.


# Set input and output paths for the model that you want to process:
MODEL_PATH = "bfm2017-1_bfm_nomouth.bin"
JSON_PATH  = "bfm2017-1_bfm_nomouth_edge_topology.json"


def generate_edge_topology(triangle_list):
	# add 1 to all indices to make 1-based indices for Matlab
	triangle_list = [sorted([v0+1, v1+1, v2+1]) for v0, v1, v2 in triangle_list]
	edge_list = []
	index_list = []
	print("triangles:", len(triangle_list))

	for v0, v1, v2 in triangle_list:
		for e in [[v0, v1], [v0, v2], [v1, v2]]:
			if e not in edge_list:
				edge_list.append(e)
				idx = None
				for i, (f0, f1, f2) in enumerate(triangle_list):
					if (f0 == e[0] and (f1 == e[1] or f2 == e[1])) or (f1 == e[0] and f2 == e[1]):
						if idx is None:
							idx = i+1
						else:
							index_list.append([idx, i+1])
							idx = None
							break
				if idx is not None:
					index_list.append([0, idx])
	print("edges:", len(edge_list))
	return index_list, edge_list


if __name__ == "__main__":
	import eos

	model = eos.morphablemodel.load_model(MODEL_PATH)
	triangle_list = model.get_shape_model().get_triangle_list()
	index_list, edge_list = generate_edge_topology(triangle_list)

	# Save it as an eos EdgeTopology in json format:
	edge_topology = eos.morphablemodel.EdgeTopology(index_list, edge_list)
	eos.morphablemodel.save_edge_topology(edge_topology, JSON_PATH)
	print("Finished generating edge-topology file and saved it as", JSON_PATH)
