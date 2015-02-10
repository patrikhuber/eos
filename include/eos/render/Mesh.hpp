/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/Mesh.hpp
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

#ifndef MESH_HPP_
#define MESH_HPP_

#include "opencv2/core/core.hpp"

#include <vector>
#include <array>
#include <string>

namespace eos {
	namespace render {

/**
 * This class represents a 3D mesh consisting of vertices and vertex color
 * information. Additionally it stores the indices that specify which vertices
 * to use to generate the triangle mesh out of the vertices.
 */
class Mesh
{
public:
	std::vector<cv::Vec4f> vertices; ///< 3D vertex positions.
	std::vector<cv::Vec3f> colors; ///< Color information for each vertex. Expected to be in RGB order.
	std::vector<cv::Vec2f> texcoords; ///< Texture coordinates for each vertex.

	std::vector<std::array<int, 3>> tvi; ///< Triangle vertex indices
	std::vector<std::array<int, 3>> tci; ///< Triangle color indices
};

/**
 * Writes an obj file of the given Mesh that can be read by e.g. Meshlab.
 *
 * @param[in] mesh The mesh to save as obj.
 * @param[in] filename Output filename.
 */
void writeObj(Mesh mesh, std::string filename);

	} /* namespace render */
} /* namespace eos */

#endif /* MESH_HPP_ */
