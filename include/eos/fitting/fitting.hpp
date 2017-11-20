/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/fitting.hpp
 *
 * Copyright 2015 Patrik Huber
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

#ifndef FITTING_HPP_
#define FITTING_HPP_

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

#include "Eigen/Core"

#include <algorithm>
#include <cassert>
#include <optional>
#include <vector>

namespace eos {
namespace fitting {

/**
 * Convenience function that fits the shape model and expression blendshapes to
 * landmarks. Makes the fitted PCA shape and blendshape coefficients accessible
 * via the out parameters \p pca_shape_coefficients and \p blendshape_coefficients.
 * It iterates PCA-shape and blendshape fitting until convergence
 * (usually it converges within 5 to 10 iterations).
 *
 * See fit_shape_model(cv::Mat, eos::morphablemodel::MorphableModel, std::vector<eos::morphablemodel::Blendshape>, std::vector<cv::Vec2f>, std::vector<int>, float lambda)
 * for a simpler overload that just returns the shape instance.
 *
 * @param[in] affine_camera_matrix The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] image_points 2D landmarks from an image to fit the model to.
 * @param[in] vertex_indices The vertex indices in the model that correspond to the 2D points.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to fit all coefficients.
 * @param[out] pca_shape_coefficients Output parameter that will contain the resulting pca shape coefficients.
 * @param[out] blendshape_coefficients Output parameter that will contain the resulting blendshape coefficients.
 * @return The fitted model shape instance.
 */
inline Eigen::VectorXf fit_shape(Eigen::Matrix<float, 3, 4> affine_camera_matrix,
                                 const morphablemodel::MorphableModel& morphable_model,
                                 const std::vector<morphablemodel::Blendshape>& blendshapes,
                                 const std::vector<Eigen::Vector2f>& image_points,
                                 const std::vector<int>& vertex_indices, float lambda,
                                 std::optional<int> num_coefficients_to_fit,
                                 std::vector<float>& pca_shape_coefficients,
                                 std::vector<float>& blendshape_coefficients)
{
    using Eigen::MatrixXf;
    using Eigen::VectorXf;

    const MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

    std::vector<float> last_blendshape_coeffs, current_blendshape_coeffs;
    std::vector<float> last_pca_coeffs, current_pca_coeffs;
    current_blendshape_coeffs.resize(blendshapes.size()); // starting values t_0, all zeros
    // no starting values for current_pca_coeffs required, since we start with the shape fitting, and cv::norm
    // of an empty vector is 0.

    VectorXf combined_shape;
    do // run at least once:
    {
        last_blendshape_coeffs = current_blendshape_coeffs;
        last_pca_coeffs = current_pca_coeffs;
        // Estimate the PCA shape coefficients with the current blendshape coefficients (0 in the first
        // iteration):
        const VectorXf mean_plus_blendshapes =
            morphable_model.get_shape_model().get_mean() +
            blendshapes_as_basis *
                Eigen::Map<const VectorXf>(last_blendshape_coeffs.data(), last_blendshape_coeffs.size());
        current_pca_coeffs = fitting::fit_shape_to_landmarks_linear(
            morphable_model.get_shape_model(), affine_camera_matrix, image_points, vertex_indices,
            mean_plus_blendshapes, lambda, num_coefficients_to_fit);

        // Estimate the blendshape coefficients with the current PCA model estimate:
        const VectorXf pca_model_shape = morphable_model.get_shape_model().draw_sample(current_pca_coeffs);
        current_blendshape_coeffs = fitting::fit_blendshapes_to_landmarks_nnls(
            blendshapes, pca_model_shape, affine_camera_matrix, image_points, vertex_indices);

        // Todo/Note: Could move next line outside the loop, not needed in here actually
        combined_shape = pca_model_shape +
                         blendshapes_as_basis * Eigen::Map<const VectorXf>(current_blendshape_coeffs.data(),
                                                                           current_blendshape_coeffs.size());
    } while (
        std::abs(Eigen::Map<const VectorXf>(current_pca_coeffs.data(), current_pca_coeffs.size()).norm() -
                 Eigen::Map<const VectorXf>(last_pca_coeffs.data(), last_pca_coeffs.size()).norm()) >= 0.01 ||
        std::abs(
            Eigen::Map<const VectorXf>(current_blendshape_coeffs.data(), current_blendshape_coeffs.size()).norm() -
            Eigen::Map<const VectorXf>(last_blendshape_coeffs.data(), last_blendshape_coeffs.size()).norm()) >= 0.01);

    pca_shape_coefficients = current_pca_coeffs;
    blendshape_coefficients = current_blendshape_coeffs;

    return combined_shape;
};

/**
 * Convenience function that fits the shape model and expression blendshapes to
 * landmarks. It iterates PCA-shape and blendshape fitting until convergence
 * (usually it converges within 5 to 10 iterations).
 *
 * @param[in] affine_camera_matrix The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] image_points 2D landmarks from an image to fit the model to.
 * @param[in] vertex_indices The vertex indices in the model that correspond to the 2D points.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to fit all coefficients.
 * @return The fitted model shape instance.
 */
inline Eigen::VectorXf fit_shape(Eigen::Matrix<float, 3, 4> affine_camera_matrix,
                                 const morphablemodel::MorphableModel& morphable_model,
                                 const std::vector<morphablemodel::Blendshape>& blendshapes,
                                 const std::vector<Eigen::Vector2f>& image_points,
                                 const std::vector<int>& vertex_indices, float lambda = 3.0f,
                                 std::optional<int> num_coefficients_to_fit = std::optional<int>())
{
    std::vector<float> unused;
    return fit_shape(affine_camera_matrix, morphable_model, blendshapes, image_points, vertex_indices, lambda,
                     num_coefficients_to_fit, unused, unused);
};

/**
 * @brief Takes a LandmarkCollection of 2D landmarks and, using the landmark_mapper, finds the
 * corresponding 3D vertex indices and returns them, along with the coordinates of the 3D points.
 *
 * The function only returns points which the landmark mapper was able to convert, and skips all
 * points for which there is no mapping. Thus, the number of returned points might be smaller than
 * the number of input points.
 * All three output vectors have the same size and contain the points in the same order.
 * \c landmarks can be an eos::core::LandmarkCollection<cv::Vec2f> or an rcr::LandmarkCollection<cv::Vec2f>.
 *
 * Notes:
 * - Split into two functions, one which maps from 2D LMs to vtx_idx and returns a reduced vec of 2D LMs.
 *   And then the other one to go from vtx_idx to a vector<Vec4f>.
 * - Place in a potentially more appropriate header (shape-fitting?).
 * - Could move to detail namespace or forward-declare.
 * - \c landmarks has to be a collection of LMs, with size(), [] and Vec2f ::coordinates.
 * - Probably model_points would better be a Vector3f and not in homogeneous coordinates?
 *
 * @param[in] landmarks A LandmarkCollection of 2D landmarks.
 * @param[in] landmark_mapper A mapper which maps the 2D landmark identifiers to 3D model vertex indices.
 * @param[in] morphable_model Model to get the 3D point coordinates from.
 * @return A tuple of [image_points, model_points, vertex_indices].
 */
template <class T>
inline auto get_corresponding_pointset(const T& landmarks, const core::LandmarkMapper& landmark_mapper,
                                       const morphablemodel::MorphableModel& morphable_model)
{
    using Eigen::Vector2f;
    using Eigen::Vector4f;
    using std::vector;

    // These will be the final 2D and 3D points used for the fitting:
    vector<Vector4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices;    // their vertex indices
    vector<Vector2f> image_points; // the corresponding 2D landmark points

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.get());
        auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
        model_points.emplace_back(Vector4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }
    return std::make_tuple(image_points, model_points, vertex_indices);
};

/**
 * @brief Concatenates two std::vector's of the same type and returns the concatenated
 * vector. The elements of the second vector are appended after the first one.
 *
 * Note: Move to detail namespace? It's used for the contour fitting, but doesn't really belong here.
 *
 * @param[in] vec_a First vector.
 * @param[in] vec_b Second vector.
 * @return The concatenated vector.
 */
template <class T>
inline auto concat(const std::vector<T>& vec_a, const std::vector<T>& vec_b)
{
    std::vector<T> concatenated_vec;
    concatenated_vec.reserve(vec_a.size() + vec_b.size());
    concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_a), std::end(vec_a));
    concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_b), std::end(vec_b));
    return concatenated_vec;
};

/**
 * @brief Fits the given expression model to landmarks.
 *
 * The function uses fit_blendshapes_to_landmarks_nnls(...) if the given expression model consists of
 * Blendshapes, and fit_shape_to_landmarks_linear(...) if it is a PCA model.
 *
 * Note that if the expression model is a PCA model, and we're doing fit_shape_to_landmarks_linear(...), we
 * should probably pass the regularisation and num_coeffs_to_fit along as well.
 *
 * @param[in] expression_model The expression model (blendshapes or a PCA model).
 * @param[in] face_instance A shape instance from which the expression coefficients should be estimated (i.e. the current mesh without expressions, e.g. estimated from a previous PCA-model fitting).
 * @param[in] affine_camera_matrix Second vector.
 * @param[in] landmarks 2D landmarks from an image to fit the blendshapes to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @return A vector of fitted expression coefficients.
 */
inline std::vector<float> fit_expressions(const morphablemodel::ExpressionModel& expression_model,
                                          const Eigen::VectorXf& face_instance,
                                          const Eigen::Matrix<float, 3, 4>& affine_camera_matrix,
                                          const std::vector<Eigen::Vector2f>& landmarks,
                                          const std::vector<int>& vertex_ids)
{
    std::vector<float> expression_coefficients;
    if (std::holds_alternative<morphablemodel::PcaModel>(expression_model))
    {
        const auto& pca_expression_model = std::get<morphablemodel::PcaModel>(expression_model);
        return fit_shape_to_landmarks_linear(pca_expression_model, affine_camera_matrix, landmarks,
                                             vertex_ids, face_instance); // lambda, num_coeffs_to_fit, ...
    } else if (std::holds_alternative<morphablemodel::Blendshapes>(expression_model))
    {
        const auto& expression_blendshapes = std::get<morphablemodel::Blendshapes>(expression_model);
        return fit_blendshapes_to_landmarks_nnls(expression_blendshapes, face_instance, affine_camera_matrix,
                                                 landmarks, vertex_ids);
    } else
    {
        throw std::runtime_error("The given expression_model doesn't contain a PcaModel or Blendshapes.");
    }
};

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
 * starting values in the fitting. When the function returns, they contain the coefficients from
 * the last iteration.
 *
 * Use render::Mesh fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, std::optional<int>, float).
 * for a simpler overload with reasonable defaults and no optional output.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * Note: If the given \p morphable_model contains a PCA expression model, alternating the shape identity and
 * expression fitting is theoretically not needed - the basis matrices could be stacked, and then both
 * coefficients could be solved for in one go. The two bases are most likely not orthogonal though.
 * In any case, alternating hopefully doesn't do any harm.
 *
 * Todo: Add a convergence criterion.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
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
 * @param[in,out] expression_coefficients If given, will be used as initial expression blendshape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[out] fitted_image_points Debug parameter: Returns all the 2D points that have been used for the fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<core::Mesh, fitting::RenderingParameters> fit_shape_and_pose(
    const morphablemodel::MorphableModel& morphable_model,
    const core::LandmarkCollection<Eigen::Vector2f>& landmarks, const core::LandmarkMapper& landmark_mapper,
    int image_width, int image_height, const morphablemodel::EdgeTopology& edge_topology,
    const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour,
    int num_iterations, std::optional<int> num_shape_coefficients_to_fit, float lambda,
    std::optional<fitting::RenderingParameters> initial_rendering_params,
    std::vector<float>& pca_shape_coefficients, std::vector<float>& expression_coefficients,
    std::vector<Eigen::Vector2f>& fitted_image_points)
{
    //assert(blendshapes.size() > 0);
    assert(landmarks.size() >= 4);
    assert(image_width > 0 && image_height > 0);
    assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
    assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());
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
    // pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!

    /*if (expression_coefficients.empty())
    {
        expression_coefficients.resize(blendshapes.size());
    }*/

    // This should really be part of MorphableModel (or BilinearModel... this only draws a shape sample).
    /*const auto draw_shape_sample = [](const morphablemodel::MorphableModel& morphable_model, std::vector<float> shape_coefficients, std::vector<float> expression_coefficients)
    {
        assert(morphable_model.has_separate_expression_model());

        Eigen::VectorXf shape_sample;
        Eigen::VectorXf expression_sample;

        if (shape_coefficients.empty())
        {
            shape_sample = morphable_model.get_shape_model().get_mean();
        }
        else
        {
            shape_sample = morphable_model.get_shape_model().draw_sample(shape_coefficients);
        }
        // Get a sample of the expression model, depending on whether it's a PcaModel or Blendshapes:
        if (morphable_model.has_separate_expression_model() && std::holds_alternative<morphablemodel::PcaModel>(morphable_model.get_expression_model().value())) {
            const auto& pca_expression_model = std::get<morphablemodel::PcaModel>(morphable_model.get_expression_model().value());
            assert(pca_expression_model.get_data_dimension() == morphable_model.get_shape_model().get_data_dimension());
            if (expression_coefficients.empty())
            {
                expression_sample = pca_expression_model.get_mean();
            }
            else
            {
                expression_sample = pca_expression_model.draw_sample(expression_coefficients);
            }
        }
        else if (morphable_model.has_separate_expression_model() && std::holds_alternative<morphablemodel::Blendshapes>(morphable_model.get_expression_model().value()))
        {
            const auto& expression_blendshapes = std::get<morphablemodel::Blendshapes>(morphable_model.get_expression_model().value());
            assert(expression_blendshapes.size() > 0);
            assert(expression_blendshapes[0].deformation.rows() == morphable_model.get_shape_model().get_data_dimension());
            if (expression_coefficients.empty())
            {
                expression_sample.setZero(expression_blendshapes[0].deformation.rows());
            }
            else
            {
                expression_sample = morphablemodel::to_matrix(expression_blendshapes) * Eigen::Map<const Eigen::VectorXf>(expression_coefficients.data(), expression_coefficients.size());
            }
        }
        else
        {
            throw std::runtime_error("This MorphableModel doesn't contain an expression model in the form of a PcaModel or Blendshapes.");
        }
        shape_sample += expression_sample;
        return shape_sample;
    };
    */

    // Note that this doesn't check whether the shape and expression model dimensions match.
    // Note: We're calling this in a loop, and morphablemodel::to_matrix(expression_blendshapes) now gets called again in every fitting iteration.
    const auto draw_expression_sample =
        [](const std::variant<morphablemodel::PcaModel, morphablemodel::Blendshapes>& expression_model,
           vector<float> expression_coefficients) {
            VectorXf expression_sample;
            // Get a sample of the expression model, depending on whether it's a PcaModel or Blendshapes:
            if (std::holds_alternative<morphablemodel::PcaModel>(expression_model))
            {
                const auto& pca_expression_model = std::get<morphablemodel::PcaModel>(expression_model);
                if (expression_coefficients.empty())
                {
                    expression_sample = pca_expression_model.get_mean();
                } else
                {
                    expression_sample = pca_expression_model.draw_sample(expression_coefficients);
                }
            } else if (std::holds_alternative<morphablemodel::Blendshapes>(expression_model))
            {
                const auto& expression_blendshapes = std::get<morphablemodel::Blendshapes>(expression_model);
                assert(expression_blendshapes.size() > 0);
                if (expression_coefficients.empty())
                {
                    expression_sample.setZero(expression_blendshapes[0].deformation.rows());
                } else
                {
                    expression_sample = morphablemodel::to_matrix(expression_blendshapes) *
                                        Eigen::Map<const VectorXf>(expression_coefficients.data(),
                                                                   expression_coefficients.size());
                }
            } else
            {
                throw std::runtime_error(
                    "The given expression_model doesn't contain a PcaModel or Blendshapes.");
            }
            return expression_sample;
        };

    // const MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

    // Current mesh - either from the given coefficients, or the mean:
    VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
    // Note we don't check whether the shape and expression model dimensions match.
    VectorXf current_combined_shape =
        current_pca_shape +
        draw_expression_sample(morphable_model.get_expression_model().value(), expression_coefficients);
    auto current_mesh = morphablemodel::sample_to_mesh(
        current_combined_shape, morphable_model.get_color_model().get_mean(),
        morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

    // The 2D and 3D point correspondences used for the fitting:
    vector<Vector4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices; // their vertex indices
    vector<Vector2f> image_points; // the corresponding 2D landmark points

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
    // and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
    for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.value());
        Vector4f vertex(current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                        current_mesh.vertices[vertex_idx][2], 1.0f);
        model_points.emplace_back(vertex);
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }

    // Need to do an initial pose fit to do the contour fitting inside the loop.
    // We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    fitting::ScaledOrthoProjectionParameters current_pose =
        fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image_height);
    fitting::RenderingParameters rendering_params(current_pose, image_width, image_height);

    const Eigen::Matrix<float, 3, 4> affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image_width, image_height);
    assert(morphable_model.has_separate_expression_model()); // Note: We could also just skip the expression fitting in this case.
    expression_coefficients = fit_expressions(morphable_model.get_expression_model().value(), current_pca_shape, affine_from_ortho, image_points, vertex_indices);

    // Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape coeffs have been given):
    current_combined_shape =
        current_pca_shape +
        draw_expression_sample(morphable_model.get_expression_model().value(), expression_coefficients);
    current_mesh = morphablemodel::sample_to_mesh(
        current_combined_shape, morphable_model.get_color_model().get_mean(),
        morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

    // The static (fixed) landmark correspondences which will stay the same throughout
    // the fitting (the inner face landmarks):
    const auto fixed_image_points = image_points;
    const auto fixed_vertex_indices = vertex_indices;

    for (int i = 0; i < num_iterations; ++i)
    {
        image_points = fixed_image_points;
        vertex_indices = fixed_vertex_indices;
        // Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
        vector<Vector2f> image_points_contour;
        vector<int> vertex_indices_contour;
        const auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params.get_rotation())[1]);
        // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
            fitting::get_contour_correspondences(landmarks, contour_landmarks, model_contour, yaw_angle,
                                                 current_mesh, rendering_params.get_modelview(),
                                                 rendering_params.get_projection(),
                                                 fitting::get_opencv_viewport(image_width, image_height));
        // Add the contour correspondences to the set of landmarks that we use for the fitting:
        vertex_indices = fitting::concat(vertex_indices, vertex_indices_contour);
        image_points = fitting::concat(image_points, image_points_contour);

        // Fit the occluding (away-facing) contour using the detected contour LMs:
        vector<Vector2f> occluding_contour_landmarks;
        if (yaw_angle >= 0.0f) // positive yaw = subject looking to the left
        { // the left contour is the occluding one we want to use ("away-facing")
            auto contour_landmarks_ =
                core::filter(landmarks, contour_landmarks.left_contour); // Can do this outside of the loop
            std::for_each(begin(contour_landmarks_), end(contour_landmarks_),
                          [&occluding_contour_landmarks](auto&& lm) {
                              occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
                          });
        } else
        {
            auto contour_landmarks_ = core::filter(landmarks, contour_landmarks.right_contour);
            std::for_each(begin(contour_landmarks_), end(contour_landmarks_),
                          [&occluding_contour_landmarks](auto&& lm) {
                              occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
                          });
        }
        const auto edge_correspondences = fitting::find_occluding_edge_correspondences(
            current_mesh, edge_topology, rendering_params, occluding_contour_landmarks, 180.0f);
        image_points = fitting::concat(image_points, edge_correspondences.first);
        vertex_indices = fitting::concat(vertex_indices, edge_correspondences.second);

        // Get the model points of the current mesh, for all correspondences that we've got:
        model_points.clear();
        for (auto v : vertex_indices)
        {
            model_points.push_back({current_mesh.vertices[v][0], current_mesh.vertices[v][1],
                                    current_mesh.vertices[v][2], 1.0f});
        }

        // Re-estimate the pose, using all correspondences:
        current_pose =
            fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image_height);
        rendering_params = fitting::RenderingParameters(current_pose, image_width, image_height);

        const Eigen::Matrix<float, 3, 4> affine_from_ortho =
            fitting::get_3x4_affine_camera_matrix(rendering_params, image_width, image_height);

        // Estimate the PCA shape coefficients with the current blendshape coefficients:
        const VectorXf mean_plus_blendshapes =
            morphable_model.get_shape_model().get_mean() +
            draw_expression_sample(morphable_model.get_expression_model().value(), expression_coefficients);
        pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear(
            morphable_model.get_shape_model(), affine_from_ortho, image_points, vertex_indices,
            mean_plus_blendshapes, lambda, num_shape_coefficients_to_fit);

        // Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
        expression_coefficients =
            fit_expressions(morphable_model.get_expression_model().value(), current_pca_shape,
                            affine_from_ortho, image_points, vertex_indices);

        current_combined_shape =
            current_pca_shape +
            draw_expression_sample(morphable_model.get_expression_model().value(), expression_coefficients);
        current_mesh = morphablemodel::sample_to_mesh(
            current_combined_shape, morphable_model.get_color_model().get_mean(),
            morphable_model.get_shape_model().get_triangle_list(),
            morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
    }

    fitted_image_points = image_points;
    return { current_mesh, rendering_params }; // I think we could also work with a Mat face_instance in this function instead of a Mesh, but it would convolute the code more (i.e. more complicated to access vertices).
};

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If you want to access the values of shape or blendshape coefficients, or want to set starting
 * values for them, use the following overload to this function:
 * std::pair<render::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, std::optional<int>, float, std::optional<fitting::RenderingParameters>, std::vector<float>&, std::vector<float>&, std::vector<cv::Vec2f>&)
 *
 * Todo: Add a convergence criterion.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
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
inline std::pair<core::Mesh, fitting::RenderingParameters>
fit_shape_and_pose(const morphablemodel::MorphableModel& morphable_model,
                   const core::LandmarkCollection<Eigen::Vector2f>& landmarks,
                   const core::LandmarkMapper& landmark_mapper, int image_width, int image_height,
                   const morphablemodel::EdgeTopology& edge_topology,
                   const fitting::ContourLandmarks& contour_landmarks,
                   const fitting::ModelContour& model_contour, int num_iterations = 5,
                   std::optional<int> num_shape_coefficients_to_fit = std::nullopt, float lambda = 50.0f)
{
    std::vector<float> pca_coeffs;
    std::vector<float> blendshape_coeffs;
    std::vector<Eigen::Vector2f> fitted_image_points;
    return fit_shape_and_pose(morphable_model, landmarks, landmark_mapper, image_width, image_height,
                              edge_topology, contour_landmarks, model_contour, num_iterations,
                              num_shape_coefficients_to_fit, lambda, std::nullopt, pca_coeffs,
                              blendshape_coeffs, fitted_image_points);
};

} /* namespace fitting */
} /* namespace eos */

#endif /* FITTING_HPP_ */
