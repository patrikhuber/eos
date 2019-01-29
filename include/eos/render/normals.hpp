/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/normals.hpp
 *
 * Copyright 2014-2019 Patrik Huber
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

#ifndef EOS_RENDER_NORMALS_HPP
#define EOS_RENDER_NORMALS_HPP

#include "glm/vec3.hpp"
#include "glm/geometric.hpp"

#include "Eigen/Core"

namespace eos {
namespace render {

/**
 * Computes the normal of a face (triangle), i.e. the per-face normal. Returned normal will be unit length.
 *
 * Assumes the triangle is given in CCW order, i.e. vertices in counterclockwise order on the screen are
 * front-facing.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return The unit-length normal of the given triangle.
 */
inline Eigen::Vector3f compute_face_normal(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1,
                                           const Eigen::Vector3f& v2)
{
    Eigen::Vector3f n = (v1 - v0).cross(v2 - v0); // v0-to-v1 x v0-to-v2
    return n.normalized();
};

/**
 * Computes the normal of a face (triangle), i.e. the per-face normal. Returned normal will be unit length.
 *
 * Assumes the triangle is given in CCW order, i.e. vertices in counterclockwise order on the screen are
 * front-facing.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return The unit-length normal of the given triangle.
 */
inline Eigen::Vector3f compute_face_normal(const Eigen::Vector4f& v0, const Eigen::Vector4f& v1,
                                           const Eigen::Vector4f& v2)
{
    Eigen::Vector4f n = (v1 - v0).cross3(v2 - v0); // v0-to-v1 x v0-to-v2
    return n.head<3>().normalized();
};

/**
 * Computes the normal of a face (triangle), i.e. the per-face normal. Returned normal will be unit length.
 *
 * Assumes the triangle is given in CCW order, i.e. vertices in counterclockwise order on the screen are
 * front-facing.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return The unit-length normal of the given triangle.
 */
inline glm::vec3 compute_face_normal(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    glm::vec3 n = glm::cross(v1 - v0, v2 - v0); // v0-to-v1 x v0-to-v2
    n = glm::normalize(n);
    return n;
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_RENDER_NORMALS_HPP */
