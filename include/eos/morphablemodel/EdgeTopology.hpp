/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/EdgeTopology.hpp
 *
 * Copyright 2016 Patrik Huber
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

#ifndef EDGETOPOLOGY_HPP_
#define EDGETOPOLOGY_HPP_

#include "cereal/cereal.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/json.hpp"

#include <array>
#include <fstream>
#include <vector>

namespace eos {
namespace morphablemodel {

/**
 * @brief A struct containing a 3D shape model's edge topology.
 *
 * This struct contains all edges of a 3D mesh, and for each edge, it
 * contains the two faces and the two vertices that are adjacent to that
 * edge. This is used in the iterated closest edge fitting (ICEF).
 *
 * Note: The indices are 1-based, so 1 needs to be subtracted before using
 * them as mesh indices. An index of 0 as first array element means that
 * it's an edge that lies on the mesh boundary, i.e. they are only
 * adjacent to one face.
 * We should explore a less error-prone way to store this data, but that's
 * how it is done in Matlab by the original code.
 *
 * adjacent_faces.size() is equal to adjacent_vertices.size().
 */
struct EdgeTopology
{
    std::vector<std::array<int, 2>>
        adjacent_faces; ///< num_edges x 2 matrix storing faces adjacent to each edge
    std::vector<std::array<int, 2>>
        adjacent_vertices; ///< num_edges x 2 matrix storing vertices adjacent to each edge

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] archive The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(adjacent_faces), CEREAL_NVP(adjacent_vertices));
    };
};

/**
 * Saves a 3DMM edge topology file to a json file.
 *
 * @param[in] edge_topology A model's edge topology.
 * @param[in] filename The file to write.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
inline void save_edge_topology(EdgeTopology edge_topology, std::string filename)
{
    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Error creating given file: " + filename);
    }
    cereal::JSONOutputArchive output_archive(file);
    output_archive(cereal::make_nvp("edge_topology", edge_topology));
};

/**
 * Load a 3DMM edge topology file from a json file.
 *
 * @param[in] filename The file to load the edge topology from.
 * @return A struct containing the edge topology.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
inline EdgeTopology load_edge_topology(std::string filename)
{
    EdgeTopology edge_topology;
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Error opening file for reading: " + filename);
    }
    cereal::JSONInputArchive output_archive(file);
    output_archive(cereal::make_nvp("edge_topology", edge_topology));

    return edge_topology;
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* EDGETOPOLOGY_HPP_ */
