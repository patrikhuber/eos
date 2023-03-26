/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp
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

#ifndef EOS_NONLINEAR_CAMERA_ESTIMATION_DETAIL_HPP
#define EOS_NONLINEAR_CAMERA_ESTIMATION_DETAIL_HPP

#include "eos/render/matrix_projection.hpp"

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <vector>

namespace eos {
namespace fitting {
namespace detail {

/**
 * @brief Projects a given point to 2D, using the given angles (in radians), translation, frustum, and image
 * size, using orthographic projection.
 *
 * Expects the landmark points to be in OpenCV convention, i.e. origin TL.
 *
 * The rotation will be applied in order yaw, pitch, roll (y, x, z).
 *
 * This, in principle, does the same as eos's project(), but it is tailored to use the parameters from
 * estimate_orthographic_camera(), and to use Euler angles, orthographic projection and an OpenCV viewport.
 */
inline Eigen::Vector3f project_ortho(Eigen::Vector3f point, float rot_x_pitch, float rot_y_yaw,
                                     float rot_z_roll, float tx, float ty, float frustum_scale,
                                     /* fixed params now: */ int width, int height)
{
    // Note: We could (should?) make the function arguments double.

    // Note: We could alternatively use quaternions (or another representation) to represent the rotation, to
    // be independent of the Euler angle order/convention. But this works too.
    const Eigen::Matrix3f rot_mtx_x = Eigen::AngleAxisf(rot_x_pitch, Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Matrix3f rot_mtx_y = Eigen::AngleAxisf(rot_y_yaw, Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Matrix3f rot_mtx_z = Eigen::AngleAxisf(rot_z_roll, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    Eigen::Matrix4f model_view_mtx = Eigen::Matrix4f::Identity();
    model_view_mtx.block<3, 3>(0, 0) = rot_mtx_z * rot_mtx_x * rot_mtx_y; // P_2d = RPY * P_3d
    model_view_mtx.col(3).head<3>() = Eigen::Vector3f(tx, ty, 0.0f);

    // Constructing an orthographic projection matrix, without n/f.
    const float aspect = static_cast<float>(width) / height;
    const auto ortho_mtx = render::ortho(-1.0f * aspect * frustum_scale, 1.0f * aspect * frustum_scale,
                                         -1.0f * frustum_scale, 1.0f * frustum_scale);
    const auto viewport = get_opencv_viewport(width, height);
    const auto res = render::project(point, model_view_mtx, ortho_mtx, viewport);
    return res;
};

/**
 * @brief Generic functor for Eigen's optimisation algorithms.
 */
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

/**
 * @brief LevenbergMarquardt cost function for the orthographic camera estimation.
 */
struct OrthographicParameterProjection : Functor<double>
{
public:
    // Creates a new OrthographicParameterProjection object with given data.
    OrthographicParameterProjection(std::vector<Eigen::Vector2f> image_points,
                                    std::vector<Eigen::Vector4f> model_points, int width, int height)
        : Functor<double>(6, image_points.size()), image_points(image_points), model_points(model_points),
          width(width), height(height){};

    // x = current params, fvec = the errors/differences of the proj with current params and the GT
    // (image_points)
    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
        const float aspect = static_cast<float>(width) / height;
        for (int i = 0; i < values(); i++)
        {
            // Eigen to glm:
            // glm::vec3 point_3d(model_points[i][0], model_points[i][1], model_points[i][2]);
            // projection given current params x:
            Eigen::Vector3f proj_with_current_param_esti =
                project_ortho(model_points[i].head<3>(), x[0], x[1], x[2], x[3], x[4], x[5], width, height);
            // Eigen::Vector2f proj_point_2d(proj_with_current_param_esti.x, proj_with_current_param_esti.y);
            // diff of current proj to ground truth, our error
            //const auto diff = (proj_point_2d - image_points[i]).norm();
            const auto diff = (proj_with_current_param_esti.head<2>() - image_points[i]).norm();
            // fvec should contain the differences
            // don't square it.
            fvec[i] = diff;
        }
        return 0;
    };

private:
    std::vector<Eigen::Vector2f> image_points;
    std::vector<Eigen::Vector4f> model_points;
    int width;
    int height;
};

} /* namespace detail */
} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_NONLINEAR_CAMERA_ESTIMATION_DETAIL_HPP */
