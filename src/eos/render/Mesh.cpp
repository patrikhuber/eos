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

	} /* namespace render */
} /* namespace eos */
