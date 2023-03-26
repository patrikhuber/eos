/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/nonlinear_camera_estimation.hpp
 *
 * Copyright 2015, 2023 Patrik Huber
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

#ifndef EOS_NONLINEAR_CAMERA_ESTIMATION_HPP
#define EOS_NONLINEAR_CAMERA_ESTIMATION_HPP

#include "eos/fitting/detail/nonlinear_camera_estimation_detail.hpp"
#include "eos/fitting/RenderingParameters.hpp"

#include "Eigen/Geometry"
#include "unsupported/Eigen/NonLinearOptimization"

#include <vector>
#include <cassert>

namespace eos {
namespace fitting {

/**
 * @brief This algorithm estimates the rotation angles and translation of the model, as
 * well as the viewing frustum of the camera, given a set of corresponding 2D-3D points.
 *
 * It assumes an orthographic camera and estimates 6 parameters,
 * [r_x, r_y, r_z, t_x, t_y, frustum_scale], where the first five describe how to transform
 * the model, and the last one describes the cameras viewing frustum (see CameraParameters).
 * This 2D-3D correspondence problem is solved using Eigen's LevenbergMarquardt algorithm.
 *
 * The method is slightly inspired by "Computer Vision: Models Learning and Inference",
 * Simon J.D. Prince, 2012, but different in a lot of respects.
 *
 * Eigen's LM implementation requires at least 6 data points, so we require >= 6 corresponding points.
 *
 * Notes/improvements:
 * The algorithm works reliable as it is, however, it could be improved with the following:
 *  - A better initial guess (see e.g. Prince)
 *  - We could add a parameter to pass an initial guess
 *  - Using the analytic derivatives instead of Eigen::NumericalDiff - they're easy to calculate.
 *
 * @param[in] image_points A list of 2D image points.
 * @param[in] model_points Corresponding points of a 3D model.
 * @param[in] width Width of the image (or viewport).
 * @param[in] height Height of the image (or viewport).
 * @return The estimated model and camera parameters.
 */
inline RenderingParameters estimate_orthographic_camera(std::vector<Eigen::Vector2f> image_points,
                                                        std::vector<Eigen::Vector4f> model_points, int width,
                                                        int height)
{
    assert(image_points.size() == model_points.size());
    assert(image_points.size() >= 6); // Number of correspondence points given needs to be equal to or larger than 6

    const float aspect = static_cast<float>(width) / height;

    // Set up the initial parameter vector and the cost function:
    int num_params = 6;
    Eigen::VectorXd parameters;              // [rot_x_pitch, rot_y_yaw, rot_z_roll, t_x, t_y, frustum_scale]
    parameters.setConstant(num_params, 0.0); // Set all 6 values to zero (except frustum_scale, see next line)
    parameters[5] = 110.0; // This is just a rough hand-chosen scaling estimate - we could do a lot better. But it works.
    detail::OrthographicParameterProjection cost_function(image_points, model_points, width, height);

    // Note: we have analytical derivatives, so we should use them!
    Eigen::NumericalDiff<detail::OrthographicParameterProjection> cost_function_with_derivative(cost_function,
                                                                                                0.0001);
    // I had to change the default value of epsfcn, it works well around 0.0001. It couldn't produce the
    // derivative with the default, I guess the changes in the gradient were too small.

    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<detail::OrthographicParameterProjection>> lm(
        cost_function_with_derivative);
    const auto info = lm.minimize(parameters); // we could or should use the return value
    // 'parameters' contains the solution now.

    Frustum camera_frustum{
        -1.0f * aspect * static_cast<float>(parameters[5]), 1.0f * aspect * static_cast<float>(parameters[5]),
        -1.0f * static_cast<float>(parameters[5]), 1.0f * static_cast<float>(parameters[5])};
    return RenderingParameters(CameraType::Orthographic, camera_frustum, static_cast<float>(parameters[0]),
                               static_cast<float>(parameters[1]), static_cast<float>(parameters[2]),
                               static_cast<float>(parameters[3]), static_cast<float>(parameters[4]), width,
                               height);
};

} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_NONLINEAR_CAMERA_ESTIMATION_HPP */
