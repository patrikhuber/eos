/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/vertex_visibility.hpp
 *
 * Copyright 2019, 2020, 2023 Patrik Huber
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

#ifndef EOS_RENDER_VERTEX_VISIBILITY_HPP
#define EOS_RENDER_VERTEX_VISIBILITY_HPP

#include "eos/render/detail/RayDirection.hpp"
#include "eos/render/ray_triangle_intersect.hpp"

#include "Eigen/Core"

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"

#include <array>
#include <vector>
#include <algorithm>

namespace eos {
namespace render {

/**
 * Computes whether the given \p probe_vertex is visible or self-occluded by the mesh formed by the given
 * vertices and triangle indices.
 *
 * The function uses simple ray casting and checks whether the ray intersects any triangle of the given mesh.
 * Thus, the algorithm can become quite slow for larger meshes.
 * Depending on \p ray_direction_type, the rays are either casted from each vertex along the positive z axis,
 * or towards the origin (0, 0, 0).
 *
 * Note/Todo: We have a bit of an unnecessary conversion between GLM and Eigen going on. Ideally, ...:
 *   - probe_vertex should rather be an Eigen::Vector3f
 *   - mesh_vertices should rather be std::vector<Eigen::Vector3f>.
 *
 * @param[in] probe_vertex A single vertex to compute the visibility for.
 * @param[in] mesh_vertices A set of vertices that form a mesh.
 * @param[in] mesh_triangle_vertex_indices Triangle indices corresponding to the given mesh.
 * @param[in] ray_direction_type Whether the occlusion should be computed under orthographic or perspective projection.
 * @return Returns whether the given vertex is visible (true if the vertex is visible, false if it is self-occluded).
 */
inline bool is_vertex_visible(const glm::vec4 probe_vertex, const std::vector<glm::vec4>& mesh_vertices,
                              const std::vector<std::array<int, 3>>& mesh_triangle_vertex_indices,
                              detail::RayDirection ray_direction_type)
{
    using glm::vec3;

    bool visible = true;
    const vec3 ray_origin(probe_vertex);
    const vec3 ray_direction = [&ray_direction_type, &probe_vertex]() {
        if (ray_direction_type == detail::RayDirection::Parallel)
        {
            // For orthographic cameras, shoot towards the user with a parallel ray:
            return vec3(0, 0, 1);
        } else
        {
            // For perspective cameras, we shoot the ray from the vertex towards the camera origin (which
            // is at (0, 0, 0)):
            return vec3(-probe_vertex);
        }
    }();

    // We check in a brute-force manner whether the ray hits any triangles, by looping through all the
    // triangles, for each vertex of the mesh. This is a very slow way to do this, of course. We should
    // better use an AABB tree.
    // For every triangle of the rotated mesh:
    for (const auto& tri : mesh_triangle_vertex_indices)
    {
        const auto& v0 = mesh_vertices[tri[0]];
        const auto& v1 = mesh_vertices[tri[1]];
        const auto& v2 = mesh_vertices[tri[2]];

        const auto intersect =
            ray_triangle_intersect(ray_origin, ray_direction, vec3(v0), vec3(v1), vec3(v2), false);
        // first is bool intersect, second is the distance t
        if (intersect.first == true)
        {
            // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the
            // intersection:
            // Check if in front or behind?
            if (intersect.second.value() <= 1e-4)
            {
                continue; // the intersection is behind the vertex, we don't care about it
            }
            // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
            visible = false;
            break;
        }
    }
    return visible;
};

/**
 * For each given vertex, compute whether it is visible or self-occluded by the mesh formed by the given
 * vertices and triangle indices.
 *
 * The function uses simple ray casting for each vertex, and checks whether each ray intersects any other
 * triangle. Thus, the algorithm can become quite slow for larger meshes.
 * Depending on \p ray_direction_type, the rays are either casted from each vertex along the positive z axis,
 * or towards the origin (0, 0, 0).
 *
 * @param[in] vertices A set of vertices that form a mesh.
 * @param[in] triangle_vertex_indices Triangle indices corresponding to the given mesh.
 * @param[in] ray_direction_type Whether the occlusion should be computed under orthographic or perspective projection.
 * @return Returns the per-vertex visibility (true if the vertex is visible, false if it is self-occluded).
 */
inline std::vector<bool>
compute_per_vertex_self_occlusion(const std::vector<Eigen::Vector3f>& vertices,
                                  const std::vector<std::array<int, 3>>& triangle_vertex_indices,
                                  detail::RayDirection ray_direction_type)
{
    using glm::vec3;
    using glm::vec4;
    using std::vector;

    // We are already given vertices in view space (or we should not transform them). We just need to do a bit
    // of a back-and-forth between Eigen and GLM here.
    vector<vec4> viewspace_vertices;
    std::for_each(std::begin(vertices), std::end(vertices), [&viewspace_vertices](const auto& v) {
        viewspace_vertices.push_back(vec4(v.x(), v.y(), v.z(), 1.0));
    });

    vector<bool> per_vertex_visibility;
    for (const auto& vertex : viewspace_vertices)
    {
        const bool visible =
            is_vertex_visible(vertex, viewspace_vertices, triangle_vertex_indices, ray_direction_type);
        per_vertex_visibility.push_back(visible);
    }
    return per_vertex_visibility;
};

/**
 * For each given vertex, compute whether it is visible or self-occluded by the mesh formed by the given
 * vertices and triangle indices.
 *
 * Transforms the vertices into view space first using the given \p modelview matrix.
 *
 * @param[in] vertices A set of vertices that form a mesh.
 * @param[in] triangle_vertex_indices Triangle indices corresponding to the given mesh.
 * @param[in] modelview Model-view matrix, to transform the given vertices from model space to view space.
 * @param[in] ray_direction_type Whether the occlusion should be computed under orthographic or perspective
 * projection.
 * @return Returns the per-vertex visibility (true if the vertex is visible, false if it is self-occluded).
 */
inline std::vector<bool>
compute_per_vertex_self_occlusion(const std::vector<Eigen::Vector3f>& vertices,
                                  const std::vector<std::array<int, 3>>& triangle_vertex_indices,
                                  const Eigen::Matrix4f& modelview, detail::RayDirection ray_direction_type)
{
    std::vector<Eigen::Vector3f> viewspace_vertices;
    std::for_each(std::begin(vertices), std::end(vertices), [&viewspace_vertices, &modelview](const auto& v) {
        const Eigen::Vector4f transformed_vertex = modelview * v.homogeneous();
        viewspace_vertices.push_back(transformed_vertex.head<3>());
    });

    return compute_per_vertex_self_occlusion(viewspace_vertices, triangle_vertex_indices, ray_direction_type);
};

} // namespace render
} // namespace eos

#endif /* EOS_RENDER_VERTEX_VISIBILITY_HPP */
