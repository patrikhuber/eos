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
#include "eos/core/Image.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/core/read_obj.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/PcaModel.hpp"
#include "eos/pca/pca.hpp"
#include "eos/render/texture_extraction.hpp"

#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_Image.hpp"

#include "eos/cpp17/optional.hpp"
#include "pybind11_optional.hpp"

#include "Eigen/Core"

#include <cassert>
#include <string>

namespace py = pybind11;
using namespace eos;

/**
 * Generate python bindings for the eos library using pybind11.
 */
PYBIND11_MODULE(eos, eos_module)
{
    eos_module.doc() = "Python bindings for the eos 3D Morphable Face Model fitting library.\n\nFor an "
                       "overview of the functionality, see the documentation of the submodules. For the full "
                       "documentation, see the C++ doxygen documentation.";

    /**
     * Bindings for the eos::core namespace:
     *  - LandmarkMapper
     *  - Mesh
     *  - write_obj(), write_textured_obj()
     */
    py::module core_module = eos_module.def_submodule("core", "Essential functions and classes to work with 3D face models and landmarks.");
    
    py::class_<core::LandmarkMapper>(core_module, "LandmarkMapper", "Represents a mapping from one kind of landmarks to a different format(e.g.model vertices).")
        .def(py::init<>(), "Constructs a new landmark mapper that performs an identity mapping, that is, its output is the same as the input.")
        .def(py::init<std::string>(), "Constructs a new landmark mapper from a file containing mappings from one set of landmark identifiers to another.", py::arg("filename"))
        .def("convert", &core::LandmarkMapper::convert, "Converts the given landmark name to the mapped name.", py::arg("landmark_name"));

    py::class_<core::Mesh>(core_module, "Mesh", "This class represents a 3D mesh consisting of vertices, vertex colour information and texture coordinates.")
        .def(py::init<>(), "Creates an empty mesh.")
        .def_readwrite("vertices", &core::Mesh::vertices, "Vertices")
        .def_readwrite("tvi", &core::Mesh::tvi, "Triangle vertex indices")
        .def_readwrite("colors", &core::Mesh::colors, "Colour data")
        .def_readwrite("tci", &core::Mesh::tci, "Triangle colour indices (usually the same as tvi)")
        .def_readwrite("texcoords", &core::Mesh::texcoords, "Texture coordinates");

    core_module.def("write_obj", &core::write_obj, "Writes the given Mesh to an obj file.", py::arg("mesh"), py::arg("filename"));
    core_module.def("write_textured_obj", &core::write_textured_obj, "Writes the given Mesh to an obj file, including texture coordinates, and an mtl file containing a reference to the isomap. The texture (isomap) has to be saved separately.", py::arg("mesh"), py::arg("filename"));

    core_module.def("read_obj", &core::read_obj, "Reads the given Wavefront .obj file into a Mesh.", py::arg("filename"));

    /**
     * Bindings for the eos::morphablemodel namespace:
     *  - Blendshape
     *  - load_blendshapes()
     */
    py::module morphablemodel_module = eos_module.def_submodule("morphablemodel", "Functionality to represent a Morphable Model, its PCA models, and functions to load models and blendshapes.");

    py::class_<morphablemodel::Blendshape>(morphablemodel_module, "Blendshape", "A class representing a 3D blendshape.")
        .def(py::init<>(), "Creates an empty blendshape.")
        .def(py::init<std::string, Eigen::VectorXf>(), "Create a blendshape with given name and deformation vector.", py::arg("name"), py::arg("deformation"))
        .def_readwrite("name", &morphablemodel::Blendshape::name, "Name of the blendshape.")
        .def_readwrite("deformation", &morphablemodel::Blendshape::deformation, "A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices. Has the same format as PcaModel::mean.");

    morphablemodel_module.def("load_blendshapes", &morphablemodel::load_blendshapes, "Load a file with blendshapes from a cereal::BinaryInputArchive (.bin) from the harddisk.", py::arg("filename"));
    morphablemodel_module.def("save_blendshapes", &morphablemodel::save_blendshapes, "Save a set of blendshapes to the harddisk as a cereal::BinaryOutputArchive (.bin).", py::arg("blendshapes"), py::arg("filename"));

    /**
     *  - PcaModel
     *  - MorphableModel
     *  - load_model(), save_model()
     *  - load_pca_model(), save_pca_model()
     */
    py::class_<morphablemodel::PcaModel>(morphablemodel_module, "PcaModel", "Class representing a PcaModel with a mean, eigenvectors and eigenvalues, as well as a list of triangles to build a mesh.")
        .def(py::init<>(), "Creates an empty model.")
        .def(py::init<Eigen::VectorXf, Eigen::MatrixXf, Eigen::VectorXf, std::vector<std::array<int, 3>>>(), "Construct a PCA model from given mean, orthonormal PCA basis, eigenvalues and triangle list.", py::arg("mean"), py::arg("orthonormal_pca_basis"), py::arg("eigenvalues"), py::arg("triangle_list"))
        .def("get_num_principal_components", &morphablemodel::PcaModel::get_num_principal_components, "Returns the number of principal components in the model.")
        .def("get_data_dimension", &morphablemodel::PcaModel::get_data_dimension, "Returns the dimension of the data, i.e. the number of shape dimensions.")
        .def("get_triangle_list", &morphablemodel::PcaModel::get_triangle_list, "Returns a list of triangles on how to assemble the vertices into a mesh.")
        .def("get_mean", &morphablemodel::PcaModel::get_mean, "Returns the mean of the model.")
        .def("get_mean_at_point", &morphablemodel::PcaModel::get_mean_at_point, "Return the value of the mean at a given vertex index.", py::arg("vertex_index"))
        .def("get_orthonormal_pca_basis", &morphablemodel::PcaModel::get_orthonormal_pca_basis, "Returns the orthonormal PCA basis matrix, i.e. the eigenvectors. Each column of the matrix is an eigenvector.")
        .def("get_rescaled_pca_basis", &morphablemodel::PcaModel::get_rescaled_pca_basis, "Returns the rescaled PCA basis matrix, i.e. the eigenvectors. Each column of the matrix is an eigenvector, and each eigenvector has been rescaled by multiplying it with the square root of its eigenvalue.")
        .def("get_eigenvalues", &morphablemodel::PcaModel::get_eigenvalues, "Returns the models eigenvalues.")
        .def("draw_sample", (Eigen::VectorXf(morphablemodel::PcaModel::*)(std::vector<float>)const)&morphablemodel::PcaModel::draw_sample, "Returns a sample from the model with the given PCA coefficients. The given coefficients should follow a standard normal distribution, i.e. not be scaled with their eigenvalues/variances.", py::arg("coefficients"));

    py::class_<morphablemodel::MorphableModel>(morphablemodel_module, "MorphableModel", "A class representing a 3D Morphable Model, consisting of a shape- and colour (albedo) PCA model, as well as texture (uv) coordinates.")
        .def(py::init<morphablemodel::PcaModel, morphablemodel::PcaModel, std::vector<std::array<double, 2>>>(), "Create a Morphable Model from a shape and a colour PCA model, and optional texture coordinates.", py::arg("shape_model"), py::arg("color_model"), py::arg("texture_coordinates") = std::vector<std::array<double, 2>>())
        .def(py::init<morphablemodel::PcaModel, morphablemodel::ExpressionModel, morphablemodel::PcaModel, std::vector<std::array<double, 2>>>(), "Create a Morphable Model from a shape and a colour PCA model, an expression PCA model or blendshapes, and optional texture coordinates.", py::arg("shape_model"), py::arg("expression_model"), py::arg("color_model"), py::arg("texture_coordinates") = std::vector<std::array<double, 2>>())
        .def("get_shape_model", &morphablemodel::MorphableModel::get_shape_model, "Returns the PCA shape model of this Morphable Model.") // Not sure if that'll really be const in Python? I think Python does a copy each time this gets called?
        .def("get_color_model", &morphablemodel::MorphableModel::get_color_model, "Returns the PCA colour (albedo) model of this Morphable Model.")
        .def("get_expression_model", &morphablemodel::MorphableModel::get_expression_model, "Returns the shape expression model or an empty optional if the Morphable Model does not have a separate expression model.")
        .def("get_mean", &morphablemodel::MorphableModel::get_mean, "Returns the mean of the shape- and colour model as a Mesh.")
        .def("draw_sample", (core::Mesh(morphablemodel::MorphableModel::*)(std::vector<float>, std::vector<float>)const)&morphablemodel::MorphableModel::draw_sample, "Returns a sample from the model with the given shape- and colour PCA coefficients.", py::arg("shape_coefficients"), py::arg("color_coefficients"))
        .def("draw_sample", (core::Mesh(morphablemodel::MorphableModel::*)(std::vector<float>, std::vector<float>, std::vector<float>)const)&morphablemodel::MorphableModel::draw_sample, "Returns a sample from the model with the given shape, expression, and colour PCA coefficients. The MorphableModel has to have an expression model, otherwise, this function will throw.", py::arg("shape_coefficients"), py::arg("expression_coefficients"), py::arg("color_coefficients"))
        .def("has_color_model", &morphablemodel::MorphableModel::has_color_model, "Returns true if this Morphable Model contains a colour model, and false if it is a shape-only model.")
        .def("has_separate_expression_model", &morphablemodel::MorphableModel::has_separate_expression_model, "Returns true if this Morphable Model contains a separate PCA or Blendshapes expression model.")
        .def("get_texture_coordinates", &morphablemodel::MorphableModel::get_texture_coordinates, "Returns the texture coordinates for all the vertices in the model.");

    morphablemodel_module.def("load_model", &morphablemodel::load_model, "Load a Morphable Model from a cereal::BinaryInputArchive (.bin) from the harddisk.", py::arg("filename"));
    morphablemodel_module.def("save_model", &morphablemodel::save_model, "Save a Morphable Model as cereal::BinaryOutputArchive.", py::arg("model"), py::arg("filename"));
    morphablemodel_module.def("load_pca_model", &morphablemodel::load_pca_model, "Load a PCA model from a cereal::BinaryInputArchive (.bin) from the harddisk.", py::arg("filename"));
    morphablemodel_module.def("save_pca_model", &morphablemodel::save_pca_model, "Save a PCA model as cereal::BinaryOutputArchive.", py::arg("model"), py::arg("filename"));

    /**
     *  - EdgeTopology
     *  - load_edge_topology()
     */
    py::class_<morphablemodel::EdgeTopology>(morphablemodel_module, "EdgeTopology", "A struct containing a 3D shape model's edge topology.")
        .def(py::init<std::vector<std::array<int, 2>>, std::vector<std::array<int, 2>>>(), "Construct a new EdgeTopology with given adjacent_faces and adjacent_vertices.", py::arg("adjacent_faces"), py::arg("adjacent_vertices")) // py::init<> uses brace-initialisation: http://pybind11.readthedocs.io/en/stable/advanced/classes.html#brace-initialization
        .def_readwrite("adjacent_faces", &morphablemodel::EdgeTopology::adjacent_faces, "A num_edges x 2 matrix storing faces adjacent to each edge.")
        .def_readwrite("adjacent_vertices", &morphablemodel::EdgeTopology::adjacent_vertices, "A num_edges x 2 matrix storing vertices adjacent to each edge.");

    morphablemodel_module.def("load_edge_topology", &morphablemodel::load_edge_topology, "Load a 3DMM edge topology file from a json file.", py::arg("filename"));
    morphablemodel_module.def("save_edge_topology", &morphablemodel::save_edge_topology, "Save a 3DMM edge topology file to a json file.", py::arg("edge_topology"), py::arg("filename"));

    /**
     * Bindings for the eos::pca namespace:
     *  - Covariance
     *  - pca()
     */
    py::module pca_module = eos_module.def_submodule("pca", "PCA and functionality to build statistical models.");

    py::enum_<pca::Covariance>(pca_module, "Covariance", "A flag specifying how to compute the covariance matrix in the PCA.")
        .value("AtA", pca::Covariance::AtA)
        .value("AAt", pca::Covariance::AAt);

    pca_module.def("pca", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXf>, pca::Covariance>(&pca::pca), "Compute PCA on a mean-centred data matrix, and return the eigenvectors and respective eigenvalues.", py::arg("data"), py::arg("covariance_type") = pca::Covariance::AtA);
    pca_module.def("pca", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXf>, int, pca::Covariance>(&pca::pca), "Performs PCA and returns num_eigenvectors_to_keep eigenvectors and eigenvalues.", py::arg("data"), py::arg("num_eigenvectors_to_keep"), py::arg("covariance_type") = pca::Covariance::AtA);
    pca_module.def("pca", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXf>, float, pca::Covariance>(&pca::pca), "Performs PCA and returns the number of eigenvectors and eigenvalues to retain 'variance_to_keep' variance of the original data.", py::arg("data"), py::arg("variance_to_keep"), py::arg("covariance_type") = pca::Covariance::AtA);
    pca_module.def("pca", py::overload_cast<const Eigen::Ref<const Eigen::MatrixXf>, std::vector<std::array<int, 3>>, pca::Covariance>(&pca::pca), "Performs PCA on the given data (including subtracting the mean) and returns the built PcaModel.", py::arg("data"), py::arg("triangle_list"), py::arg("covariance_type") = pca::Covariance::AtA);

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
        .def_property_readonly("R",
             [](const fitting::ScaledOrthoProjectionParameters& p) {
            Eigen::Matrix3f R; // we could probably use Eigen::Map
            for (int col = 0; col < 4; ++col)
                for (int row = 0; row < 4; ++row)
                    R(row, col) = p.R[col][row];
            return R;
    }, "Rotation matrix") // we can easily make this writable if ever required, just need to add a lambda function with the Eigen to glm matrix conversion.
        .def_readwrite("s", &fitting::ScaledOrthoProjectionParameters::s, "Scale")
        .def_readwrite("tx", &fitting::ScaledOrthoProjectionParameters::tx, "x translation")
        .def_readwrite("ty", &fitting::ScaledOrthoProjectionParameters::ty, "y translation");

    py::class_<fitting::RenderingParameters>(fitting_module, "RenderingParameters", "Represents a set of estimated model parameters (rotation, translation) and camera parameters (viewing frustum).")
        .def(py::init<fitting::ScaledOrthoProjectionParameters, int, int>(), "Create a RenderingParameters object from an instance of estimated ScaledOrthoProjectionParameters.")
        .def("get_rotation",
             [](const fitting::RenderingParameters& p) {
                 return Eigen::Vector4f(p.get_rotation().x, p.get_rotation().y, p.get_rotation().z, p.get_rotation().w);
             },
             "Returns the rotation quaternion [x y z w].")
        .def("get_rotation_euler_angles",
             [](const fitting::RenderingParameters& p) {
                 const glm::vec3 euler_angles = glm::eulerAngles(p.get_rotation());
                 return Eigen::Vector3f(euler_angles[0], euler_angles[1], euler_angles[2]);
            },
             "Returns the rotation's Euler angles (in radians) as [pitch, yaw, roll].")
        .def("get_modelview",
            [](const fitting::RenderingParameters& p) {
                Eigen::Matrix4f model_view; // we could probably use Eigen::Map
                for (int col = 0; col < 4; ++col)
                    for (int row = 0; row < 4; ++row)
                        model_view(row, col) = p.get_modelview()[col][row];
                return model_view;
            },
            "Returns the 4x4 model-view matrix.")
        .def("get_projection",
            [](const fitting::RenderingParameters& p) {
                Eigen::Matrix4f projection; // we could probably use Eigen::Map
                for (int col = 0; col < 4; ++col)
                    for (int row = 0; row < 4; ++row)
                        projection(row, col) = p.get_projection()[col][row];
                return projection;
            }, "Returns the 4x4 projection matrix.");

    fitting_module.def("estimate_orthographic_projection_linear", &fitting::estimate_orthographic_projection_linear,
                       "This algorithm estimates the parameters of a scaled orthographic projection, given a set of corresponding 2D-3D points.",
                       py::arg("image_points"), py::arg("model_points"), py::arg("is_viewport_upsidedown") = py::none(), py::arg("viewport_height") = 0);

    py::class_<fitting::ContourLandmarks>(fitting_module, "ContourLandmarks", "Defines which 2D landmarks comprise the right and left face contour.")
        .def_static("load", &fitting::ContourLandmarks::load, "Helper method to load contour landmarks from a text file with landmark mappings, like ibug_to_sfm.txt.", py::arg("filename"));

    py::class_<fitting::ModelContour>(fitting_module, "ModelContour", "Definition of the vertex indices that define the right and left model contour.")
        .def_static("load", &fitting::ModelContour::load, "Helper method to load a ModelContour from a json file from the hard drive.", py::arg("filename"));

    fitting_module.def("fit_shape_and_pose",
        [](const morphablemodel::MorphableModel& morphable_model,
           const std::vector<morphablemodel::Blendshape>& blendshapes,
           const std::vector<Eigen::Vector2f>& landmarks, const std::vector<std::string>& landmark_ids,
           const core::LandmarkMapper& landmark_mapper, int image_width, int image_height,
           const morphablemodel::EdgeTopology& edge_topology,
           const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour,
           int num_iterations, cpp17::optional<int> num_shape_coefficients_to_fit, float lambda) {
            assert(landmarks.size() == landmark_ids.size());
            std::vector<float> pca_coeffs;
            std::vector<float> blendshape_coeffs;
            std::vector<Eigen::Vector2f> fitted_image_points;
            core::LandmarkCollection<Eigen::Vector2f> landmark_collection;
            for (int i = 0; i < landmarks.size(); ++i)
            {
                landmark_collection.push_back(core::Landmark<Eigen::Vector2f>{ landmark_ids[i], Eigen::Vector2f(landmarks[i][0], landmarks[i][1]) });
            }
            auto result = fitting::fit_shape_and_pose(morphable_model, landmark_collection, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda, cpp17::nullopt, pca_coeffs, blendshape_coeffs, fitted_image_points);
            return std::make_tuple(result.first, result.second, pca_coeffs, blendshape_coeffs);
        },
        "Fit the pose (camera), shape model, and expression blendshapes to landmarks, in an iterative way. Returns a tuple (mesh, rendering_parameters, shape_coefficients, blendshape_coefficients).",
        py::arg("morphable_model"), py::arg("blendshapes"), py::arg("landmarks"), py::arg("landmark_ids"), py::arg("landmark_mapper"), py::arg("image_width"), py::arg("image_height"), py::arg("edge_topology"), py::arg("contour_landmarks"), py::arg("model_contour"), py::arg("num_iterations") = 5, py::arg("num_shape_coefficients_to_fit") = py::none(), py::arg("lambda") = 30.0f);

    /**
     * Bindings for the eos::render namespace:
     *  - extract_texture()
     */
    py::module render_module = eos_module.def_submodule("render", "3D mesh and texture extraction functionality.");

    render_module.def("extract_texture",
                      [](const core::Mesh& mesh, const fitting::RenderingParameters& rendering_params,
                         const core::Image3u& image, bool compute_view_angle, int isomap_resolution) {
                          Eigen::Matrix<float, 3, 4> affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
                          return render::extract_texture(mesh, affine_from_ortho, image, compute_view_angle, render::TextureInterpolation::NearestNeighbour, isomap_resolution);
                      },
                      "Extracts the texture of the face from the given image and stores it as isomap (a rectangular texture map).",
                      py::arg("mesh"), py::arg("rendering_params"), py::arg("image"), py::arg("compute_view_angle") = false, py::arg("isomap_resolution") = 512);
};
