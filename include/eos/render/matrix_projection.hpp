/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/matrix_projection.hpp
 *
 * Copyright 2023 Patrik Huber
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

#ifndef EOS_MATRIX_PROJECTION_HPP
#define EOS_MATRIX_PROJECTION_HPP

#include "Eigen/Core"

#include <cmath>

namespace eos {
namespace render {

/**
 * Creates a matrix for a right-handed, symmetric perspective-view frustum.
 *
 * The function follows the OpenGL clip volume definition, which is also the GLM default. The near and far
 * clip planes correspond to z normalized device coordinates of -1 and +1 respectively.
 *
 * This function is equivalent to glm::perspectiveRH_NO(...).
 *
 * More details can be found on the gluPerspective man page:
 * https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml.
 *
 * @param[in] fov_y Specifies the field of view angle in the y direction. Expressed in radians.
 * @param[in] aspect Specifies the aspect ratio that determines the field of view in the x direction. The
 * aspect ratio is the ratio of x (width) to y (height).
 * @param[in] z_near Specifies the distance from the viewer to the near clipping plane (always positive).
 * @param[in] z_far Specifies the distance from the viewer to the far clipping plane (always positive).
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return The corresponding perspective projection matrix.
 */
template <typename T>
Eigen::Matrix4<T> perspective(T fov_y, T aspect, T z_near, T z_far)
{
    // Will this assert work? std::abs probably won't work on T?
    // assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

    // Note: We want to use std::tan() for floating point types, and for users not using Ceres. However when
    // using the function with Ceres's auto-diff, we need to use ceres::tan(). We might need an extra "using
    // ceres::tan" here.
    using std::tan;
    const T tan_half_fov_y = tan(fov_y / static_cast<T>(2));

    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Zero(); // Note: Zero() is correct.
    result(0, 0) = static_cast<T>(1) / (aspect * tan_half_fov_y);
    result(1, 1) = static_cast<T>(1) / (tan_half_fov_y);
    result(2, 2) = -(z_far + z_near) / (z_far - z_near);
    result(3, 2) = -static_cast<T>(1);
    result(2, 3) = -(static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
    return result;
}

/**
 * @brief Creates a 2D orthographic projection matrix.
 *
 * This function sets up a two-dimensional orthographic viewing region. This is equivalent to calling glOrtho
 * with near=-1 and far=1. The function is equivalent to glm::orthoRH_NO(), but with near=-1 and far=1.
 *
 * More details can be found on the gluOrtho2D man page:
 * https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluOrtho2D.xml.
 *
 * @param[in] left Specifies the coordinates for the left vertical clipping plane.
 * @param[in] right Specifies the coordinates for the right vertical clipping plane.
 * @param[in] bottom Specifies the coordinates for the bottom horizontal clipping plane.
 * @param[in] top Specifies the coordinates for the top horizontal clipping plane.
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return The corresponding orthographic projection matrix.
 */
template <typename T>
Eigen::Matrix4<T> ortho(T left, T right, T bottom, T top)
{
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Identity();
    result(0, 0) = static_cast<T>(2) / (right - left);
    result(1, 1) = static_cast<T>(2) / (top - bottom);
    result(2, 2) = -static_cast<T>(1);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    return result;
}

/**
 * @brief Creates a matrix for an orthographic parallel viewing volume, using right-handed coordinates.
 *
 * The function follows the OpenGL clip volume definition, which is also the GLM default. The near and far
 * clip planes correspond to z normalized device coordinates of -1 and +1 respectively.
 *
 * The function is equivalent to glm::orthoRH_NO(...).
 *
 * @param[in] left Specifies the coordinates for the left vertical clipping plane.
 * @param[in] right Specifies the coordinates for the right vertical clipping plane.
 * @param[in] bottom Specifies the coordinates for the bottom horizontal clipping plane.
 * @param[in] top Specifies the coordinates for the top horizontal clipping plane.
 * @param[in] z_near Specifies the distance from the viewer to the near clipping plane (always positive).
 * @param[in] z_far Specifies the distance from the viewer to the far clipping plane (always positive).
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return The corresponding orthographic projection matrix.
 */
template <typename T>
Eigen::Matrix4<T> ortho(T left, T right, T bottom, T top, T z_near, T z_far)
{
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Identity();
    result(0, 0) = static_cast<T>(2) / (right - left);
    result(1, 1) = static_cast<T>(2) / (top - bottom);
    result(2, 2) = -static_cast<T>(2) / (z_far - z_near);
    result(0, 3) = -(right + left) / (right - left);
    result(1, 3) = -(top + bottom) / (top - bottom);
    result(2, 3) = -(z_far + z_near) / (z_far - z_near);
    return result;
};

/**
 * Project the given point_3d (from object coordinates) into window coordinates.
 *
 * The function follows the OpenGL clip volume definition. The near and far clip planes correspond to
 * z normalized device coordinates of -1 and +1 respectively.
 * This function is equivalent to glm::projectNO(...).
 *
 * More details can be found on the gluProject man page:
 * https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluProject.xml.
 *
 * @param[in] point_3d A 3D point in object coordinates.
 * @param[in] modelview_matrix A model-view matrix, transforming the point into view (camera) space.
 * @param[in] projection_matrix The projection matrix to be used.
 * @param[in] viewport The viewport transformation to be used.
 * @tparam T A floating-point scalar type, ceres::Jet, or similar compatible type.
 * @return Return the computed window coordinates.
 */
template <typename T>
Eigen::Vector3<T> project(const Eigen::Vector3<T>& point_3d, const Eigen::Matrix4<T>& modelview_matrix,
                          const Eigen::Matrix4<T>& projection_matrix, const Eigen::Vector4<T>& viewport)
{
    Eigen::Vector4<T> projected_point = projection_matrix * modelview_matrix * point_3d.homogeneous();
    projected_point /= projected_point.w();
    projected_point =
        projected_point * static_cast<T>(0.5) +
        Eigen::Vector4<T>(static_cast<T>(0.5), static_cast<T>(0.5), static_cast<T>(0.5), static_cast<T>(0.5));
    projected_point.x() = projected_point.x() * T(viewport(2)) + T(viewport(0));
    projected_point.y() = projected_point.y() * T(viewport(3)) + T(viewport(1));

    return projected_point.head<3>();
}

} /* namespace render */
} /* namespace eos */

#endif /* EOS_MATRIX_PROJECTION_HPP */
