/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/read_obj.hpp
 *
 * Copyright 2017, 2018 Patrik Huber
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

#ifndef EOS_READ_OBJ_HPP
#define EOS_READ_OBJ_HPP

#include "eos/core/Mesh.hpp"

#include "Eigen/Core"

#include <cassert>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <optional>

namespace eos {
namespace core {

namespace detail {

/**
 * @brief Splits a string into a list of tokens.
 *
 * Taken from: https://stackoverflow.com/a/1493195/1345959
 */
template <class ContainerType>
void tokenize(const std::string& str, ContainerType& tokens, const std::string& delimiters = " ",
              bool trim_empty = false)
{
    std::string::size_type pos, last_pos = 0;
    const auto length = str.length();

    using value_type = typename ContainerType::value_type;
    using size_type = typename ContainerType::size_type;

    while (last_pos < length + 1)
    {
        pos = str.find_first_of(delimiters, last_pos);
        if (pos == std::string::npos)
        {
            pos = length;
        }

        if (pos != last_pos || !trim_empty)
            tokens.push_back(value_type(str.data() + last_pos, (size_type)pos - last_pos));

        last_pos = pos + 1;
    }
};

/**
 * @brief Parse a line starting with 'v ' from an obj file.
 *
 * Can contain either 3 ('x y z') or 6 ('x y z r g b') numbers.
 * The 'v ' should have already been stripped from \p line.
 *
 * Notes:
 * Could actually potentially consist of 3, 4, 6 or 7 elements? (xyz, xyzw,
 * xyzrgb, xyzwrgb)
 * assimp only deals with 3, 4 or 6 elements:
 * https://github.com/assimp/assimp/blob/master/code/ObjFileParser.cpp#L138
 *
 * For now, we don't support obj's containing homogeneous coordinates.
 * What we could do is that if we do encounter homogeneous coords, we could just divide by 'w', like assimp
 * does.
 *
 * Another obj parser we can check: https://github.com/qnzhou/PyMesh/blob/master/src/IO/OBJParser.cpp (and
 * same file with .h)
 */
inline std::pair<Eigen::Vector3f, std::optional<Eigen::Vector3f>> parse_vertex(const std::string& line)
{
    std::vector<std::string> tokens;
    tokenize(line, tokens, " ", true); // compress multiple (and leading/trailing) whitespaces
    if (tokens.size() != 3 && tokens.size() != 6)
    {
        throw std::runtime_error(
            "Encountered a vertex ('v') line that does not consist of either 3 ('x y z') "
            "or 6 ('x y z r g b') numbers.");
    }
    const Eigen::Vector3f vertex(std::stof(tokens[0]), std::stof(tokens[1]), std::stof(tokens[2]));
    std::optional<Eigen::Vector3f> vertex_color;
    if (tokens.size() == 6)
    {
        vertex_color = Eigen::Vector3f(std::stof(tokens[3]), std::stof(tokens[4]), std::stof(tokens[5]));
    }
    return {vertex, vertex_color};
};

/**
 * @brief Parse a line starting with 'vt ' from an obj file.
 *
 * A 'vt' line can contain either two ('u v') or three ('u v w') numbers. This parser currently only supports
 * 'u v'.
 */
inline Eigen::Vector2f parse_texcoords(const std::string& line)
{
    std::vector<std::string> tokens;
    tokenize(line, tokens, " ");

    if (tokens.size() != 2)
    {
        if (tokens.size() == 3 && std::stof(tokens[2]) == 0.f)
        {
            // we support this case too - 'u v 0'
        } else
        {
            throw std::runtime_error(
                "Encountered a texture coordinates ('vt') line that does not consist of two "
                "('u v') numbers. 'u v w' is only supported in the special case if w is 0.");
        }
    }
    const Eigen::Vector2f texcoords(std::stof(tokens[0]), std::stof(tokens[1]));
    return texcoords;
};

/**
 * @brief Parse a line starting with 'vn ' from an obj file.
 *
 * This is currently not implemented yet, as eos::core::Mesh doesn't store vertex normals anyway.
 */
inline void parse_vertex_normal(const std::string& line)
{
    throw std::runtime_error("Parsing \"vn\" is not yet implemented.");
};

/**
 * @brief Parse a line starting with 'f ' from an obj file.
 *
 * Note: Indices in obj's start at 1, not at 0.
 *
 * A face line contains 3 entries for triangles, but can be quads (=4 entries), and potentially more.
 * They're triplets of: vertex, texture and normal indices. Some of them can be missing. E.g.:
 *  f 1 2 3
 *  f 3/1 4/2 5/3
 *  f 6/4/1 3/5/3 7/6/5
 *  f 7//1 8//2 9//3
 */
inline auto parse_face(const std::string& line)
{
    using std::string;
    using std::vector;

    // Obj indices are 1-based.
    vector<int> vertex_indices;  // size() = 3 or 4
    vector<int> texture_indices; // size() = 3 or 4
    vector<int> normal_indices;  // size() = 3 or 4

    vector<string> tokens;
    tokenize(line, tokens, " ", true); // compress multiple (and leading/trailing) whitespaces
    if (tokens.size() != 3 && tokens.size() != 4)
    {
        // For now we need this to be 3 (triangles) or 4 (quads)
        throw std::runtime_error("Encountered a faces ('f') line that does not consist of three or four "
                                 "blocks of numbers. We currently only support 3 blocks (triangle meshes) "
                                 "and 4 blocks (quad meshes).");
    }
    // Now for each of these tokens, we want to split on "/":
    for (const auto& token : tokens)
    {
        vector<string> subtokens;
        tokenize(token, subtokens, "/", false); // don't trim empty -if we do, we lose positional information.
        assert(subtokens.size() > 0 && subtokens.size() <= 3);

        // There should always be a vertex index:
        vertex_indices.push_back(std::stoi(subtokens[0]) - 1); // obj indices are 1-based, so we subtract one.

        if (subtokens.size() == 2)
        {
            // We've got texture coordinate indices as well:
            texture_indices.push_back(std::stoi(subtokens[1]) - 1);
        }

        if (subtokens.size() == 3)
        {
            // We've got either texcoord indices *and* normal indices, or just normal indices:
            if (!subtokens[1].empty())
            {
                // We do have texcoord indices
                texture_indices.push_back(std::stoi(subtokens[1]) - 1);
            }
            // And push_back the normal indices:
            normal_indices.push_back(std::stoi(subtokens[2]) - 1);
            // Note if we use normal indices in the future: It looked like they can be zero in some cases?
        }
    }

    return std::make_tuple(vertex_indices, texture_indices, normal_indices);
};

} /* namespace detail */

/**
 * @brief Reads the given Wavefront .obj file into a \c Mesh.
 *
 * https://en.wikipedia.org/wiki/Wavefront_.obj_file as of 22 August 2017.
 *
 * @param[in] filename Input filename (ending in ".obj").
 * @return A Mesh with the data read from the obj file.
 */
inline Mesh read_obj(std::string filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error(std::string("Could not open obj file: " + filename));
    }

    // We'll need these helper functions for the parsing:
    const auto starts_with = [](const std::string& input, const std::string& match) {
        return input.size() >= match.size() && std::equal(match.begin(), match.end(), input.begin());
    };

    /*    auto trim_left = [](const std::string& input, std::string pattern = " \t") {
            auto first = input.find_first_not_of(pattern);
            if (first == std::string::npos)
            {
                return input;
            }
            return input.substr(first, input.size());
        }; */

    Mesh mesh;

    std::string line;
    while (getline(file, line))
    {
        if (starts_with(line, "#"))
        {
            continue;
        }

        if (starts_with(line, "v "))
        { // matching with a space so that it doesn't match 'vt'
            const auto vertex_data =
                detail::parse_vertex(line.substr(2)); // pass the string without the first two characters
            mesh.vertices.push_back(vertex_data.first);
            if (vertex_data.second)
            { // there are vertex colours:
                mesh.colors.push_back(vertex_data.second.value());
            }
        }
        if (starts_with(line, "vt "))
        {
            const auto texcoords = detail::parse_texcoords(line.substr(3));
            mesh.texcoords.push_back(texcoords);
        }
        if (starts_with(line, "vn "))
        {
            // detail::parse_vertex_normal(line.substr(3));
            // Not handled yet, our Mesh class doesn't contain normals right now anyway.
        }
        // There's other things like "vp ", which we don't handle
        if (starts_with(line, "f "))
        {
            const auto face_data = detail::parse_face(line.substr(2));
            if (std::get<0>(face_data).size() == 3) // 3 triangle indices, nothing to do:
            {
                mesh.tvi.push_back(
                    {std::get<0>(face_data)[0], std::get<0>(face_data)[1], std::get<0>(face_data)[2]});
            }
            // If their sizes are 4, we convert the quad to two triangles:
            // Note: I think MeshLab does the same, it shows the number of "Faces" as twice the "f" entries
            // in the obj.
            else if (std::get<0>(face_data).size() == 4)
            {
                // Just create two faces with (quad[0], quad[1], quad[2]) and (quad[0], quad[2], quad[3]).
                mesh.tvi.push_back(
                    {std::get<0>(face_data)[0], std::get<0>(face_data)[1], std::get<0>(face_data)[2]});
                mesh.tvi.push_back(
                    {std::get<0>(face_data)[0], std::get<0>(face_data)[2], std::get<0>(face_data)[3]});
            }
            // We don't handle normal_indices for now.
        }
        // There can be other stuff in obj's like materials, named objects, etc., which are not handled here.
    }
    return mesh;
}

} /* namespace core */
} /* namespace eos */

#endif /* EOS_READ_OBJ_HPP */
