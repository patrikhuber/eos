/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/multi_image_fitting.hpp
 *
 * Copyright 2017, 2018 Patrik Huber, Philipp Kopp
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
#pragma once

#ifndef MULTI_IMAGE_FITTING_HPP_
#define MULTI_IMAGE_FITTING_HPP_

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/blendshape_fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

#include <algorithm>
#include <cassert>
#include <vector>

namespace eos {
namespace fitting {

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way. This function takes a set of images and landmarks and estimates
 * per-frame pose and expressions, as well as identity shape jointly from all images.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
 * starting values in the fitting. When the function returns, they contain the coefficients from
 * the last iteration.
 *
 * Use render::Mesh fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<Eigen::Vector2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, cpp17::optional<int>, float).
 * for a simpler overload with reasonable defaults and no optional output.
 *
 * \p num_iterations: Results are good for even a few iterations. For full convergence of all parameters,
 * it can take up to 300 iterations. In tracking, particularly if initialising with the previous frame,
 * it works well with as low as 1 to 5 iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * Todo: Add a convergence criterion.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] initial_rendering_params Currently ignored (not used).
 * @param[in,out] pca_shape_coefficients If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[in,out] blendshape_coefficients If given, will be used as initial expression blendshape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[out] fitted_image_points Debug parameter: Returns all the 2D points that have been used for the fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose(
    const morphablemodel::MorphableModel& morphable_model,
    const std::vector<morphablemodel::Blendshape>& blendshapes,
    const std::vector<core::LandmarkCollection<Eigen::Vector2f>>& landmarks,
    const core::LandmarkMapper& landmark_mapper, std::vector<int> image_width, std::vector<int> image_height,
    const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks,
    const fitting::ModelContour& model_contour, int num_iterations,
    cpp17::optional<int> num_shape_coefficients_to_fit, float lambda,
    cpp17::optional<fitting::RenderingParameters> initial_rendering_params,
    std::vector<float>& pca_shape_coefficients, std::vector<std::vector<float>>& blendshape_coefficients,
    std::vector<std::vector<Eigen::Vector2f>>& fitted_image_points)
{
    assert(blendshapes.size() > 0);
    assert(landmarks.size() > 0 && landmarks.size() == image_width.size() &&
           image_width.size() == image_height.size());
    assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
    assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());
    int num_images = static_cast<int>(landmarks.size());
    for (int j = 0; j < num_images; ++j)
    {
        assert(landmarks[j].size() >= 4);
        assert(image_width[j] > 0 && image_height[j] > 0);
    }
    // More asserts I forgot?

    using Eigen::MatrixXf;
    using Eigen::Vector2f;
    using Eigen::Vector4f;
    using Eigen::VectorXf;
    using std::vector;

    if (!num_shape_coefficients_to_fit)
    {
        num_shape_coefficients_to_fit = morphable_model.get_shape_model().get_num_principal_components();
    }

    if (pca_shape_coefficients.empty())
    {
        pca_shape_coefficients.resize(num_shape_coefficients_to_fit.value());
    }
    // Todo: This leaves the following case open: num_coeffs given is empty or defined, but the
    // pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle &
    // document!

    if (blendshape_coefficients.empty())
    {
        for (int j = 0; j < num_images; ++j)
        {
            std::vector<float> current_blendshape_coefficients;
            current_blendshape_coefficients.resize(blendshapes.size());
            blendshape_coefficients.push_back(current_blendshape_coefficients);
        }
    }

    MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

    // Current mesh - either from the given coefficients, or the mean:
    VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
    vector<VectorXf> current_combined_shapes;
    vector<core::Mesh> current_meshes;
    for (int j = 0; j < num_images; ++j)
    {
        VectorXf current_combined_shape =
            current_pca_shape +
            blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
                                                                     blendshape_coefficients[j].size());
        current_combined_shapes.push_back(current_combined_shape);

        core::Mesh current_mesh = morphablemodel::sample_to_mesh(
            current_combined_shape, morphable_model.get_color_model().get_mean(),
            morphable_model.get_shape_model().get_triangle_list(),
            morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        current_meshes.push_back(current_mesh);
    }

    // The 2D and 3D point correspondences used for the fitting:
    vector<vector<Vector4f>> model_points; // the points in the 3D shape model of all frames
    vector<vector<int>> vertex_indices;    // their vertex indices of all frames
    vector<vector<Vector2f>> image_points; // the corresponding 2D landmark points of all frames

    for (int j = 0; j < num_images; ++j)
    {
        vector<Vector4f> current_model_points;
        vector<int> current_vertex_indices;
        vector<Vector2f> current_image_points;

        // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
        // and get the corresponding model points (mean if given no initial coeffs, from the computed shape
        // otherwise):
        for (int i = 0; i < landmarks[j].size(); ++i)
        {
            const auto converted_name = landmark_mapper.convert(landmarks[j][i].name);
            if (!converted_name)
            { // no mapping defined for the current landmark
                continue;
            }
            const int vertex_idx = std::stoi(converted_name.value());
            current_model_points.emplace_back(current_meshes[j].vertices[vertex_idx].homogeneous());
            current_vertex_indices.emplace_back(vertex_idx);
            current_image_points.emplace_back(landmarks[j][i].coordinates);
        }

        model_points.push_back(current_model_points);
        vertex_indices.push_back(current_vertex_indices);
        image_points.push_back(current_image_points);
    }

    // Need to do an initial pose fit to do the contour fitting inside the loop.
    // We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    vector<fitting::RenderingParameters> rendering_params;
    for (int j = 0; j < num_images; ++j)
    {
        fitting::ScaledOrthoProjectionParameters current_pose =
            fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true,
                                                             image_height[j]);
        fitting::RenderingParameters current_rendering_params(current_pose, image_width[j], image_height[j]);
        rendering_params.push_back(current_rendering_params);

        const Eigen::Matrix<float, 3, 4> affine_from_ortho =
            fitting::get_3x4_affine_camera_matrix(current_rendering_params, image_width[j], image_height[j]);
        blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
            blendshapes, current_pca_shape, affine_from_ortho, image_points[j], vertex_indices[j]);

        // Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial
        // blendshape coeffs have been given):
        current_combined_shapes[j] =
            current_pca_shape + morphablemodel::to_matrix(blendshapes) *
                                    Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
                                                                      blendshape_coefficients[j].size());
        current_meshes[j] = morphablemodel::sample_to_mesh(
            current_combined_shapes[j], morphable_model.get_color_model().get_mean(),
            morphable_model.get_shape_model().get_triangle_list(),
            morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
    }

    // The static (fixed) landmark correspondences which will stay the same throughout
    // the fitting (the inner face landmarks):
    vector<vector<int>> fixed_vertex_indices(vertex_indices);
    vector<vector<Vector2f>> fixed_image_points(image_points);

    for (int i = 0; i < num_iterations; ++i)
    {
        vector<Eigen::Matrix<float, 3, 4>> affine_from_orthos;
        vector<VectorXf> mean_plus_blendshapes;

        image_points = fixed_image_points;
        vertex_indices = fixed_vertex_indices;

        for (int j = 0; j < num_images; ++j)
        {
            // Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
            vector<Vector2f> image_points_contour;
            vector<int> vertex_indices_contour;
            auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params[j].get_rotation())[1]);
            // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
            std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
                fitting::get_contour_correspondences(
                    landmarks[j], contour_landmarks, model_contour, yaw_angle, current_meshes[j],
                    rendering_params[j].get_modelview(), rendering_params[j].get_projection(),
                    fitting::get_opencv_viewport(image_width[j], image_height[j]));
            // Add the contour correspondences to the set of landmarks that we use for the fitting:
            vertex_indices[j] = fitting::concat(vertex_indices[j], vertex_indices_contour);
            image_points[j] = fitting::concat(image_points[j], image_points_contour);

            // Fit the occluding (away-facing) contour using the detected contour LMs:
            vector<Vector2f> occluding_contour_landmarks;
            if (yaw_angle >= 0.0f) // positive yaw = subject looking to the left
            {                      // the left contour is the occluding one we want to use ("away-facing")
                auto contour_landmarks_ = core::filter(
                    landmarks[j], contour_landmarks.left_contour); // Can do this outside of the loop
                std::for_each(
                    begin(contour_landmarks_), end(contour_landmarks_),
                    [&occluding_contour_landmarks](auto&& lm) {
                        occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
                    });
            } else
            {
                auto contour_landmarks_ = core::filter(landmarks[j], contour_landmarks.right_contour);
                std::for_each(
                    begin(contour_landmarks_), end(contour_landmarks_),
                    [&occluding_contour_landmarks](auto&& lm) {
                        occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
                    });
            }
            auto edge_correspondences = fitting::find_occluding_edge_correspondences(
                current_meshes[j], edge_topology, rendering_params[j], occluding_contour_landmarks, 180.0f);
            image_points[j] = fitting::concat(image_points[j], edge_correspondences.first);
            vertex_indices[j] = fitting::concat(vertex_indices[j], edge_correspondences.second);

            // Get the model points of the current mesh, for all correspondences that we've got:
            model_points[j].clear();
            for (auto v : vertex_indices[j])
            {
                model_points[j].push_back({current_meshes[j].vertices[v][0], current_meshes[j].vertices[v][1],
                                           current_meshes[j].vertices[v][2], 1.0f});
            }

            // Re-estimate the pose, using all correspondences:
            fitting::ScaledOrthoProjectionParameters current_pose =
                fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true,
                                                                 image_height[j]);
            rendering_params[j] = fitting::RenderingParameters(current_pose, image_width[j], image_height[j]);

            const Eigen::Matrix<float, 3, 4> affine_from_ortho =
                fitting::get_3x4_affine_camera_matrix(rendering_params[j], image_width[j], image_height[j]);
            affine_from_orthos.push_back(affine_from_ortho);

            // Estimate the PCA shape coefficients with the current blendshape coefficients:
            VectorXf current_mean_plus_blendshapes =
                morphable_model.get_shape_model().get_mean() +
                blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
                                                                         blendshape_coefficients[j].size());
            mean_plus_blendshapes.push_back(current_mean_plus_blendshapes);
        }
        pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear_multi(
            morphable_model.get_shape_model(), affine_from_orthos, image_points, vertex_indices,
            mean_plus_blendshapes, lambda, num_shape_coefficients_to_fit);

        // Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

        for (int j = 0; j < num_images; ++j)
        {
            blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
                blendshapes, current_pca_shape, affine_from_orthos[j], image_points[j], vertex_indices[j]);

            current_combined_shapes[j] =
                current_pca_shape +
                blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
                                                                         blendshape_coefficients[j].size());
            current_meshes[j] = morphablemodel::sample_to_mesh(
                current_combined_shapes[j], morphable_model.get_color_model().get_mean(),
                morphable_model.get_shape_model().get_triangle_list(),
                morphable_model.get_color_model().get_triangle_list(),
                morphable_model.get_texture_coordinates());
        }
    }

    fitted_image_points = image_points;
    return {current_meshes, rendering_params}; // I think we could also work with a Mat face_instance in this
                                              // function instead of a Mesh, but it would convolute the code
                                              // more (i.e. more complicated to access vertices).
};

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way. This function takes a set of images and landmarks and estimates
 * per-frame pose and expressions, as well as identity shape jointly from all images.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
 * starting values in the fitting. When the function returns, they contain the coefficients from
 * the last iteration.
 *
 * If you want to access the values of shape or blendshape coefficients, or want to set starting
 * values for them, use the following overload to this function:
 * std::pair<render::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<Eigen::Vector2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, cpp17::optional<int>, float, cpp17::optional<fitting::RenderingParameters>, std::vector<float>&, std::vector<float>&, std::vector<Eigen::Vector2f>&)
 *
 * \p num_iterations: Results are good for even a few iterations. For full convergence of all parameters,
 * it can take up to 300 iterations. In tracking, particularly if initialising with the previous frame,
 * it works well with as low as 1 to 5 iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>>
fit_shape_and_pose(const morphablemodel::MorphableModel& morphable_model,
                   const std::vector<morphablemodel::Blendshape>& blendshapes,
                   const std::vector<core::LandmarkCollection<Eigen::Vector2f>>& landmarks,
                   const core::LandmarkMapper& landmark_mapper, std::vector<int> image_width,
                   std::vector<int> image_height, const morphablemodel::EdgeTopology& edge_topology,
                   const fitting::ContourLandmarks& contour_landmarks,
                   const fitting::ModelContour& model_contour, int num_iterations = 5,
                   cpp17::optional<int> num_shape_coefficients_to_fit = cpp17::nullopt, float lambda = 30.0f)
{
    using std::vector;
    vector<float> pca_shape_coefficients;
    vector<vector<float>> blendshape_coefficients;
    vector<vector<Eigen::Vector2f>> fitted_image_points;

    return fit_shape_and_pose(morphable_model, blendshapes, landmarks, landmark_mapper, image_width,
                              image_height, edge_topology, contour_landmarks, model_contour, num_iterations,
                              num_shape_coefficients_to_fit, lambda, cpp17::nullopt, pca_shape_coefficients,
                              blendshape_coefficients, fitted_image_points);
};

} /* namespace fitting */
} /* namespace eos */

#endif /* MULTI_IMAGE_FITTING_HPP_ */
