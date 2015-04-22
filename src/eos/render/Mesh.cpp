/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: src/eos/render/Mesh.cpp
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
#include "eos/render/Mesh.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

#include <cassert>
#include <fstream>

namespace eos {
	namespace render {

void write_obj(Mesh mesh, std::string filename)
{
	assert(mesh.vertices.size() == mesh.colors.size());
	
	std::ofstream obj_file(filename);
	
	for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
		obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " " << mesh.colors[i][2] << " " << std::endl;
	}

	for (auto&& v : mesh.tvi) {
		// Add one because obj starts counting triangle indices at 1
		obj_file << "f " << v[0] + 1 << " " << v[1] + 1 << " " << v[2] + 1 << std::endl;
	}

	return;
}

void write_textured_obj(Mesh mesh, std::string filename)
{
	assert(mesh.vertices.size() == mesh.colors.size() && !mesh.texcoords.empty());

	std::ofstream obj_file(filename);
	
	boost::filesystem::path mtl_filename(filename);
	mtl_filename.replace_extension(".mtl");

	obj_file << "mtllib " << mtl_filename.string() << std::endl; // first line of the obj file

	for (std::size_t i = 0; i < mesh.vertices.size(); ++i) { //  same as in write_obj()
		obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " " << mesh.colors[i][2] << " " << std::endl;
	}

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
	mtl_file << "map_Kd " << texture_filename.string() << std::endl;

	return;
}

	} /* namespace render */
} /* namespace eos */
