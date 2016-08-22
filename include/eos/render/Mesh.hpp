/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
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

#include "boost/filesystem/path.hpp"

#include <vector>
#include <array>
#include <string>
#include <cassert>
#include <fstream>

namespace eos {
	namespace render {

/**
 * @brief This class represents a 3D mesh consisting of vertices, vertex colour
 * information and texture coordinates.
 *
 * Additionally it stores the indices that specify which vertices
 * to use to generate the triangle mesh out of the vertices.
 */
struct Mesh
{
	std::vector<cv::Vec4f> vertices; ///< 3D vertex positions.
	std::vector<cv::Vec3f> colors; ///< Colour information for each vertex. Expected to be in RGB order.
	std::vector<cv::Vec2f> texcoords; ///< Texture coordinates for each vertex.

	std::vector<std::array<int, 3>> tvi; ///< Triangle vertex indices
	std::vector<std::array<int, 3>> tci; ///< Triangle color indices
};

/**
 * @brief Writes the given Mesh to an obj file that for example can be read by MeshLab.
 *
 * If the mesh contains vertex colour information, it will be written to the obj as well.
 *
 * @param[in] mesh The mesh to save as obj.
 * @param[in] filename Output filename (including ".obj").
 */
inline void write_obj(Mesh mesh, std::string filename)
{
	assert(mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty());

	std::ofstream obj_file(filename);

	if (mesh.colors.empty()) {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << std::endl;
		}
	}
	else {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " " << mesh.colors[i][2] << " " << std::endl;
		}
	}

	for (auto&& v : mesh.tvi) {
		// Add one because obj starts counting triangle indices at 1
		obj_file << "f " << v[0] + 1 << " " << v[1] + 1 << " " << v[2] + 1 << std::endl;
	}

	return;
}

/**
 * @brief Writes an obj file of the given Mesh, including texture coordinates,
 * and an mtl file containing a reference to the isomap.
 *
 * The obj will contain texture coordinates for the mesh, and the
 * mtl file will link to a file named <filename>.isomap.png.
 * Note that the texture (isomap) has to be saved separately.
 *
 * @param[in] mesh The mesh to save as obj.
 * @param[in] filename Output filename, including .obj.
 */
inline void write_textured_obj(Mesh mesh, std::string filename)
{
	assert((mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty()) && !mesh.texcoords.empty());

	std::ofstream obj_file(filename);

	boost::filesystem::path mtl_filename(filename);
	mtl_filename.replace_extension(".mtl");

	obj_file << "mtllib " << mtl_filename.filename().string() << std::endl; // first line of the obj file

	// same as in write_obj():
	if (mesh.colors.empty()) {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << std::endl;
		}
	}
	else {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " " << mesh.colors[i][2] << " " << std::endl;
		}
	}
	// end

	for (std::size_t i = 0; i < mesh.texcoords.size(); ++i) {
		obj_file << "vt " << mesh.texcoords[i][0] << " " << 1.0f - mesh.texcoords[i][1] << std::endl;
		// We invert y because Meshlab's uv origin (0, 0) is on the bottom-left
	}

	obj_file << "usemtl FaceTexture" << std::endl; // the name of our texture (material) will be 'FaceTexture'

	for (auto&& v : mesh.tvi) {
		// This assumes mesh.texcoords.size() == mesh.vertices.size(). The texture indices could theoretically be different (for example in the cube-mapped 3D scan)
		// Add one because obj starts counting triangle indices at 1
		obj_file << "f " << v[0] + 1 << "/" << v[0] + 1 << " " << v[1] + 1 << "/" << v[1] + 1 << " " << v[2] + 1 << "/" << v[2] + 1 << std::endl;
	}

	std::ofstream mtl_file(mtl_filename.string());
	boost::filesystem::path texture_filename(filename);
	texture_filename.replace_extension(".isomap.png");

	mtl_file << "newmtl FaceTexture" << std::endl;
	mtl_file << "map_Kd " << texture_filename.filename().string() << std::endl;

	return;
};

	} /* namespace render */
} /* namespace eos */

#endif /* MESH_HPP_ */
