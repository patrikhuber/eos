/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/write_obj.hpp
 *
 * Copyright 2017-2019 Patrik Huber
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

#ifndef EOS_WRITE_OBJ_HPP
#define EOS_WRITE_OBJ_HPP

#include "eos/core/Mesh.hpp"

#include <cassert>
#include <fstream>
#include <string>
#include <stdexcept>

namespace eos {
namespace core {

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

    if (mesh.colors.empty())
    {
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " "
                     << mesh.vertices[i][2] << std::endl;
        }
    } else
    {
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " "
                     << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " "
                     << mesh.colors[i][2] << std::endl;
        }
    }

    if (!mesh.texcoords.empty())
    {
        for (auto&& tc : mesh.texcoords)
        {
            obj_file << "vt " << tc[0] << " " << 1.0f - tc[1] << std::endl;
            // We invert y because MeshLab's uv origin (0, 0) is on the bottom-left
        }
    }

    for (auto&& v : mesh.tvi)
    {
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

    if (filename.at(filename.size() - 4) != '.')
    {
        throw std::runtime_error(
            "Error in given filename: Expected a dot and a 3-letter extension at the end (i.e. '.obj'). " +
            filename);
    }

    // Takes a full path to a file and returns only the filename:
    const auto get_filename = [](const std::string& path) {
        auto last_slash = path.find_last_of("/\\");
        if (last_slash == std::string::npos)
        {
            return path;
        }
        return path.substr(last_slash + 1, path.size());
    };

    std::ofstream obj_file(filename);

    std::string mtl_filename(filename);
    // replace '.obj' at the end with '.mtl':
    mtl_filename.replace(std::end(mtl_filename) - 4, std::end(mtl_filename), ".mtl");

    obj_file << "mtllib " << get_filename(mtl_filename) << std::endl; // first line of the obj file

    // same as in write_obj():
    if (mesh.colors.empty())
    {
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " "
                     << mesh.vertices[i][2] << " " << std::endl;
        }
    } else
    {
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " "
                     << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " "
                     << mesh.colors[i][2] << " " << std::endl;
        }
    }
    // end

    for (std::size_t i = 0; i < mesh.texcoords.size(); ++i)
    {
        obj_file << "vt " << mesh.texcoords[i][0] << " " << 1.0f - mesh.texcoords[i][1] << std::endl;
        // We invert y because Meshlab's uv origin (0, 0) is on the bottom-left
    }

    obj_file << "usemtl FaceTexture" << std::endl; // the name of our texture (material) will be 'FaceTexture'

    for (auto&& v : mesh.tvi)
    {
        // This assumes mesh.texcoords.size() == mesh.vertices.size(). The texture indices could theoretically be different (for example in the cube-mapped 3D scan).
        // Add one because obj starts counting triangle indices at 1
        obj_file << "f " << v[0] + 1 << "/" << v[0] + 1 << " " << v[1] + 1 << "/" << v[1] + 1 << " "
                 << v[2] + 1 << "/" << v[2] + 1 << std::endl;
    }

    std::ofstream mtl_file(mtl_filename);
    std::string texture_filename(filename);
    // replace '.obj' at the end with '.isomap.png':
    texture_filename.replace(std::end(texture_filename) - 4, std::end(texture_filename), ".isomap.png");

    mtl_file << "newmtl FaceTexture" << std::endl;
    mtl_file << "map_Kd " << get_filename(texture_filename) << std::endl;

    return;
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_WRITE_OBJ_HPP */
