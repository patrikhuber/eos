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

namespace eos {
namespace render {

/**
 * Creates a matrix for a right-handed, symmetric perspective-view frustrum.
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

    const T tan_half_fov_y = tan(fov_y / static_cast<T>(2)); // ceres::tan?

    // Maybe construct with static_cast<T>(0)? => No, doesn't have c'tor.
    // Could do Eigen::Matrix4<T> result = {{1, 2, 3, 4}, {1, 2, 3, 4}...} I think.
    Eigen::Matrix4<T> result = Eigen::Matrix4<T>::Zero();
    result(0, 0) = static_cast<T>(1) / (aspect * tan_half_fov_y);
    result(1, 1) = static_cast<T>(1) / (tan_half_fov_y);
    result(2, 2) = -(z_far + z_near) / (z_far - z_near);
    result(2, 3) = -static_cast<T>(1);
    result(3, 2) = -(static_cast<T>(2) * z_far * z_near) / (z_far - z_near);
    return result;
}

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
