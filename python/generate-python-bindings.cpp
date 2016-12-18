/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: python/generate-python-bindings.cpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/PcaModel.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/render/Mesh.hpp"
#include "eos/render/texture_extraction.hpp"

#include "opencv2/core/core.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
//#include "pybind11/eigen.h"
#include "pybind11_glm.hpp"
#include "pybind11_opencv.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cassert>

namespace py = pybind11;
using namespace eos;

/**
 * Generate python bindings for the eos library using pybind11.
 */
PYBIND11_PLUGIN(eos) {
    py::module eos_module("eos", "Python bindings for the eos 3D Morphable Face Model fitting library.\n\nFor an overview of the functionality, see the documentation of the submodules. For the full documentation, see the C++ doxygen documentation.");

	/**
	 * Bindings for the eos::core namespace:
	 *  - LandmarkMapper
	 */
	py::module core_module = eos_module.def_submodule("core", "Essential functions and classes to work with 3D face models and landmarks.");
	py::class_<core::LandmarkMapper>(core_module, "LandmarkMapper", "Represents a mapping from one kind of landmarks to a different format(e.g.model vertices).")
		.def(py::init<>(), "Constructs a new landmark mapper that performs an identity mapping, that is, its output is the same as the input.")
		.def("__init__", [](core::LandmarkMapper& instance, std::string filename) { // wrap the fs::path c'tor with std::string
				new (&instance) core::LandmarkMapper(filename);
			}, "Constructs a new landmark mapper from a file containing mappings from one set of landmark identifiers to another.")
		// We can't expose the convert member function yet - need std::optional (or some trick with self/this and a lambda)
		;

	/**
	 * Bindings for the eos::morphablemodel namespace:
	 *  - PcaModel
	 *  - MorphableModel
	 *  - load_model()
	 */
	py::module morphablemodel_module = eos_module.def_submodule("morphablemodel", "Functionality to represent a Morphable Model, its PCA models, and functions to load models and blendshapes.");

	py::class_<morphablemodel::PcaModel>(morphablemodel_module, "PcaModel", "Class representing a PcaModel with a mean, eigenvectors and eigenvalues, as well as a list of triangles to build a mesh.")
		.def("get_num_principal_components", &morphablemodel::PcaModel::get_num_principal_components, "Returns the number of principal components in the model.")
		.def("get_data_dimension", &morphablemodel::PcaModel::get_data_dimension, "Returns the dimension of the data, i.e. the number of shape dimensions.")
		.def("get_triangle_list", &morphablemodel::PcaModel::get_triangle_list, "Returns a list of triangles on how to assemble the vertices into a mesh.")
		.def("get_mean", &morphablemodel::PcaModel::get_mean, "Returns the mean of the model.")
		.def("get_mean_at_point", &morphablemodel::PcaModel::get_mean_at_point, "Return the value of the mean at a given vertex index.", py::arg("vertex_index"))
		.def("draw_sample", (cv::Mat (morphablemodel::PcaModel::*)(std::vector<float>) const)&morphablemodel::PcaModel::draw_sample, "Returns a sample from the model with the given PCA coefficients. The given coefficients should follow a standard normal distribution, i.e. not be scaled with their eigenvalues/variances.", py::arg("coefficients"))
		;

	py::class_<morphablemodel::MorphableModel>(morphablemodel_module, "MorphableModel", "A class representing a 3D Morphable Model, consisting of a shape- and colour (albedo) PCA model, as well as texture (uv) coordinates.")
		.def("get_shape_model", [](const morphablemodel::MorphableModel& m) { return m.get_shape_model(); }, "Returns the PCA shape model of this Morphable Model.") // Not sure if that'll really be const in Python? I think Python does a copy each time this gets called?
		.def("get_color_model", [](const morphablemodel::MorphableModel& m) { return m.get_color_model(); }, "Returns the PCA colour (albedo) model of this Morphable Model.")
		;

	morphablemodel_module.def("load_model", &morphablemodel::load_model, "Load a Morphable Model from a cereal::BinaryInputArchive (.bin) from the harddisk.");

	/**
	 *  - Blendshape
	 *  - load_blendshapes()
	 */
	py::class_<morphablemodel::Blendshape>(morphablemodel_module, "Blendshape", "A class representing a 3D blendshape.")
		.def_readwrite("name", &morphablemodel::Blendshape::name, "Name of the blendshape.")
		.def_readwrite("deformation", &morphablemodel::Blendshape::deformation, "A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices. Has the same format as PcaModel::mean.")
		;

	morphablemodel_module.def("load_blendshapes", &morphablemodel::load_blendshapes, "Load a file with blendshapes from a cereal::BinaryInputArchive (.bin) from the harddisk.");

	/**
	 *  - EdgeTopology
	 *  - load_edge_topology()
	 */
	py::class_<morphablemodel::EdgeTopology>(morphablemodel_module, "EdgeTopology", "A struct containing a 3D shape model's edge topology.");

	morphablemodel_module.def("load_edge_topology", &morphablemodel::load_edge_topology, "Load a 3DMM edge topology file from a json file.");

	/**
	 * Bindings for the eos::render namespace:
	 * (Note: Defining Mesh before using it below in fitting::fit_shape_and_pose)
	 *  - Mesh
	 */
	py::module render_module = eos_module.def_submodule("render", "3D mesh and texture extraction functionality.");

	py::class_<render::Mesh>(render_module, "Mesh", "This class represents a 3D mesh consisting of vertices, vertex colour information and texture coordinates.")
		.def_readwrite("vertices", &render::Mesh::vertices, "Vertices")
		.def_readwrite("tvi", &render::Mesh::tvi, "Triangle vertex indices")
		.def_readwrite("colors", &render::Mesh::colors, "Colour data")
		.def_readwrite("tci", &render::Mesh::tci, "Triangle colour indices (usually the same as tvi)")
		.def_readwrite("texcoords", &render::Mesh::texcoords, "Texture coordinates")
		;

	/**
	 * Bindings for the eos::fitting namespace:
	 *  - ScaledOrthoProjectionParameters
	 *  - RenderingParameters
	 *  - estimate_orthographic_projection_linear()
	 *  - ContourLandmarks
	 *  - ModelContour
	 *  - fit_shape_and_pose()
	 */
	py::module fitting_module = eos_module.def_submodule("fitting", "Pose and shape fitting of a 3D Morphable Model.");

	py::class_<fitting::ScaledOrthoProjectionParameters>(fitting_module, "ScaledOrthoProjectionParameters", "Parameters of an estimated scaled orthographic projection.")
		.def_readwrite("R", &fitting::ScaledOrthoProjectionParameters::R, "Rotation matrix")
		.def_readwrite("s", &fitting::ScaledOrthoProjectionParameters::s, "Scale")
		.def_readwrite("tx", &fitting::ScaledOrthoProjectionParameters::tx, "x translation")
		.def_readwrite("ty", &fitting::ScaledOrthoProjectionParameters::ty, "y translation")
		;

	py::class_<fitting::RenderingParameters>(fitting_module, "RenderingParameters", "Represents a set of estimated model parameters (rotation, translation) and camera parameters (viewing frustum).")
		.def(py::init<fitting::ScaledOrthoProjectionParameters, int, int>(), "Create a RenderingParameters object from an instance of estimated ScaledOrthoProjectionParameters.")
		.def("get_rotation", [](const fitting::RenderingParameters& p) { return glm::vec4(p.get_rotation().x, p.get_rotation().y, p.get_rotation().z, p.get_rotation().w); }, "Returns the rotation quaternion [x y z w].")
		.def("get_rotation_euler_angles", [](const fitting::RenderingParameters& p) { return glm::eulerAngles(p.get_rotation()); }, "Returns the rotation's Euler angles (in radians) as [pitch, yaw, roll].")
		.def("get_modelview", &fitting::RenderingParameters::get_modelview, "Returns the 4x4 model-view matrix.")
		.def("get_projection", &fitting::RenderingParameters::get_projection, "Returns the 4x4 projection matrix.")
		;

	fitting_module.def("estimate_orthographic_projection_linear", [](std::vector<cv::Vec2f> image_points, std::vector<cv::Vec4f> model_points, bool is_viewport_upsidedown, int viewport_height) {
			const boost::optional<int> viewport_height_opt = viewport_height == 0 ? boost::none : boost::optional<int>(viewport_height);
			return fitting::estimate_orthographic_projection_linear(image_points, model_points, is_viewport_upsidedown, viewport_height_opt);
		}, "This algorithm estimates the parameters of a scaled orthographic projection, given a set of corresponding 2D-3D points.", py::arg("image_points"), py::arg("model_points"), py::arg("is_viewport_upsidedown"), py::arg("viewport_height") = 0)
		;

	py::class_<fitting::ContourLandmarks>(fitting_module, "ContourLandmarks", "Defines which 2D landmarks comprise the right and left face contour.")
		.def_static("load", &fitting::ContourLandmarks::load, "Helper method to load contour landmarks from a text file with landmark mappings, like ibug2did.txt.")
		;

	py::class_<fitting::ModelContour>(fitting_module, "ModelContour", "Definition of the vertex indices that define the right and left model contour.")
		.def_static("load", &fitting::ModelContour::load, "Helper method to load a ModelContour from a json file from the hard drive.")
		;
	
	fitting_module.def("fit_shape_and_pose", [](const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<glm::vec2>& landmarks, const std::vector<std::string>& landmark_ids, const core::LandmarkMapper& landmark_mapper, int image_width, int image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations, int num_shape_coefficients_to_fit, float lambda) {
			assert(landmarks.size() == landmark_ids.size());
			std::vector<float> pca_coeffs;
			std::vector<float> blendshape_coeffs;
			std::vector<cv::Vec2f> fitted_image_points;
			// We can change this to std::optional as soon as we switch to VS2017 and pybind supports std::optional
			const boost::optional<int> num_shape_coefficients_opt = num_shape_coefficients_to_fit == -1 ? boost::none : boost::optional<int>(num_shape_coefficients_to_fit);
			core::LandmarkCollection<cv::Vec2f> landmark_collection;
			for (int i = 0; i < landmarks.size(); ++i)
			{
				landmark_collection.push_back(core::Landmark<cv::Vec2f>{ landmark_ids[i], cv::Vec2f(landmarks[i].x, landmarks[i].y) });
			}
			auto result = fitting::fit_shape_and_pose(morphable_model, blendshapes, landmark_collection, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_opt, lambda, boost::none, pca_coeffs, blendshape_coeffs, fitted_image_points);
			return std::make_tuple(result.first, result.second, pca_coeffs, blendshape_coeffs);
		}, "Fit the pose (camera), shape model, and expression blendshapes to landmarks, in an iterative way. Returns a tuple (mesh, rendering_parameters, shape_coefficients, blendshape_coefficients).", py::arg("morphable_model"), py::arg("blendshapes"), py::arg("landmarks"), py::arg("landmark_ids"), py::arg("landmark_mapper"), py::arg("image_width"), py::arg("image_height"), py::arg("edge_topology"), py::arg("contour_landmarks"), py::arg("model_contour"), py::arg("num_iterations") = 5, py::arg("num_shape_coefficients_to_fit") = -1, py::arg("lambda") = 30.0f)
		;

	/**
	 * Bindings for the eos::render namespace:
	 * (Note: Defining down here because we need fitting::RenderingParameters to be already exposed)
	 *  - extract_texture()
	 */
	render_module.def("extract_texture", [](const render::Mesh& mesh, const fitting::RenderingParameters& rendering_params, cv::Mat image, bool compute_view_angle, int isomap_resolution) {
		cv::Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
		return render::extract_texture(mesh, affine_from_ortho, image, compute_view_angle, render::TextureInterpolation::NearestNeighbour, isomap_resolution);
	}, "Extracts the texture of the face from the given image and stores it as isomap (a rectangular texture map).", py::arg("mesh"), py::arg("rendering_params"), py::arg("image"), py::arg("compute_view_angle") = false, py::arg("isomap_resolution") = 512);

    return eos_module.ptr();
};
