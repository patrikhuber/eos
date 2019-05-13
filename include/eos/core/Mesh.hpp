/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Mesh.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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

#ifndef EOS_MESH_HPP
#define EOS_MESH_HPP

#include "Eigen/Core"

#include <array>
#include <vector>

namespace eos {
namespace core {

/**
 * @brief This class represents a 3D mesh consisting of vertices, vertex colour
 * information and texture coordinates.
 *
 * Additionally it stores the indices that specify which vertices
 * to use to generate the triangle mesh out of the vertices.
 *
 * \c texcoords should either be the same size as \c vertices (i.e. one set of texture coordinates per
 * vertex), or alternatively \c tti can be set, then a separate triangulation for the texture coordinates can
 * be used (e.g. for texture maps that contain seams).
 */
struct Mesh
{
    std::vector<Eigen::Vector3f> vertices;  ///< 3D vertex positions.
    std::vector<Eigen::Vector3f> colors;    ///< Colour information for each vertex. Expected to be in RGB order.
    std::vector<Eigen::Vector2f> texcoords; ///< Texture coordinates.

    std::vector<std::array<int, 3>> tvi;    ///< Triangle vertex indices
    std::vector<std::array<int, 3>> tci;    ///< Triangle color indices (usually the same as tvi)
    std::vector<std::array<int, 3>> tti;    ///< Triangle texture indices
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_MESH_HPP */
