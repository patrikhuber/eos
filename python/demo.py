import eos
import numpy as np

def main():
    """Demo for running the eos fitting from Python."""
    landmarks = read_pts('../bin/data/image_0010.pts')
    landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
    image_width = 1280 # Make sure to adjust these when using your own images!
    image_height = 1024

    model = eos.morphablemodel.load_model("../share/sfm_shape_3448.bin")
    blendshapes = eos.morphablemodel.load_blendshapes("../share/expression_blendshapes_3448.bin")
    landmark_mapper = eos.core.LandmarkMapper('../share/ibug_to_sfm.txt')
    edge_topology = eos.morphablemodel.load_edge_topology('../share/sfm_3448_edge_topology.json')
    contour_landmarks = eos.fitting.ContourLandmarks.load('../share/ibug_to_sfm.txt')
    model_contour = eos.fitting.ModelContour.load('../share/sfm_model_contours.json')

    (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
        landmarks, landmark_ids, landmark_mapper,
        image_width, image_height, edge_topology, contour_landmarks, model_contour)

    # Now you can use your favourite plotting/rendering library to display the fitted mesh, using the rendering
    # parameters in the 'pose' variable.

    # Or for example extract the texture map, like this:
    # import cv2
    # image = cv2.imread('../bin/data/image_0010.png')
    # isomap = eos.render.extract_texture(mesh, pose, image)


def read_pts(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])

    return landmarks

if __name__ == "__main__":
    main()
