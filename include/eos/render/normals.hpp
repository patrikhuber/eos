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

#include <vector>

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

/**
* Computes the per-face (per-triangle) normals of all triangles of the given mesh. Returned normals will be unit length.
*
* Assumes triangles are given in CCW order, i.e. vertices in counterclockwise order on the screen are front-facing.
*
* @param[in] vertices A list of vertices.
* @param[in] triangle_vertex_indices Triangle list for the given vertices.
* @return The unit-length per-face normals.
*/
inline std::vector<Eigen::Vector3f>
compute_face_normals(const std::vector<Eigen::Vector3f>& vertices,
                     const std::vector<std::array<int, 3>>& triangle_vertex_indices)
{
    std::vector<Eigen::Vector3f> face_normals;
    for (const auto& tvi : triangle_vertex_indices)
    {
        const auto face_normal = compute_face_normal(vertices[tvi[0]], vertices[tvi[1]], vertices[tvi[2]]);
        face_normals.push_back(face_normal);
    }
    return face_normals;
};

/**
 * Computes the per-vertex normals of all vertices of the given mesh. Returned normals will be unit length.
 *
 * Assumes triangles are given in CCW order, i.e. vertices in counterclockwise order on the screen are
 * front-facing.
 *
 * @param[in] vertices A list of vertices.
 * @param[in] triangle_vertex_indices Triangle list for the given vertices.
 * @param[in] face_normals Per-face normals for all triangles.
 * @return The unit-length per-vertex normals.
 */
inline std::vector<Eigen::Vector3f>
compute_vertex_normals(const std::vector<Eigen::Vector3f>& vertices,
                       const std::vector<std::array<int, 3>>& triangle_vertex_indices,
                       const std::vector<Eigen::Vector3f>& face_normals)
{
    std::vector<Eigen::Vector3f> per_vertex_normals;
    // Initialise with zeros:
    for (int i = 0; i < vertices.size(); ++i)
    {
        per_vertex_normals.emplace_back(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    }

    // Loop over the faces again:
    for (int i = 0; i < triangle_vertex_indices.size(); ++i)
    {
        const auto& tvi = triangle_vertex_indices[i];
        // Throw normal at each corner:
        per_vertex_normals[tvi[0]] += face_normals[i];
        per_vertex_normals[tvi[1]] += face_normals[i];
        per_vertex_normals[tvi[2]] += face_normals[i];
    }

    // Take average via normalization:
    for (auto& n : per_vertex_normals)
    {
        n.normalize();
    }
    return per_vertex_normals;
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_RENDER_NORMALS_HPP */
