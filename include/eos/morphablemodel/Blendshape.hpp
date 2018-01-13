/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/Blendshape.hpp
 *
 * Copyright 2015 Patrik Huber
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

#ifndef BLENDSHAPE_HPP_
#define BLENDSHAPE_HPP_

#include "cereal/archives/binary.hpp"
#include "cereal/types/string.hpp"
#include "eos/morphablemodel/io/eigen_cerealisation.hpp"

#include "Eigen/Core"

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

namespace eos {
namespace morphablemodel {

/**
 * @brief A class representing a 3D blendshape.
 *
 * A blendshape is a vector of offsets that transform the vertices of
 * a given mesh or shape instance. Usually, a blendshape is associated
 * with a deformation like a particular facial expression or a phoneme.
 */
struct Blendshape
{
    std::string name;            ///< Name of the blendshape.
    Eigen::VectorXf deformation; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of
                                 ///< model-vertices. Has the same format as PcaModel::mean.

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] archive The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(name), CEREAL_NVP(deformation));
    };
};

/**
 * Shorthand notation for an std::vector<Blendshape>.
 */
using Blendshapes = std::vector<Blendshape>;

/**
 * Helper method to load a file with blendshapes from
 * a cereal::BinaryInputArchive from the harddisk.
 *
 * @param[in] filename Filename to a blendshapes-file.
 * @return The loaded blendshapes.
 * @throw std::runtime_error When the file given in \c filename fails to be opened (most likely because the
 * file doesn't exist).
 */
inline std::vector<Blendshape> load_blendshapes(std::string filename)
{
    std::vector<Blendshape> blendshapes;

    std::ifstream file(filename, std::ios::binary);
    if (file.fail())
    {
        throw std::runtime_error("Error opening given file: " + filename);
    }
    cereal::BinaryInputArchive input_archive(file);
    input_archive(blendshapes);

    return blendshapes;
};

/**
 * Helper method to save a set of blendshapes to the
 * harddisk as a cereal::BinaryOutputArchive.
 *
 * @param[in] blendshapes The blendshapes to be saved.
 * @param[in] filename Filename for the blendshapes.
 */
inline void save_blendshapes(const std::vector<Blendshape>& blendshapes, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(blendshapes);
};

/**
 * @brief Copies the blendshapes into a matrix, with each column being a blendshape.
 *
 * @param[in] blendshapes Vector of blendshapes.
 * @return The resulting matrix.
 */
inline Eigen::MatrixXf to_matrix(const std::vector<Blendshape>& blendshapes)
{
    assert(blendshapes.size() > 0);
    // Todo: Assert all blendshapes have to have the same number of rows, and one col

    Eigen::MatrixXf blendshapes_as_basis(blendshapes[0].deformation.rows(), blendshapes.size());
    for (int i = 0; i < blendshapes.size(); ++i)
    {
        blendshapes_as_basis.col(i) = blendshapes[i].deformation;
    }
    return blendshapes_as_basis;
};

/**
 * @brief Maps an std::vector of coefficients with Eigen::Map, so it can be multiplied
 * with a blendshapes matrix.
 *
 * Note that  the resulting Eigen::Map only lives as long as the data given lives and is in scope.
 *
 * @param[in] coefficients Vector of blendshape coefficients.
 * @return An Eigen::Map pointing to the given coefficients data.
 */
inline Eigen::Map<const Eigen::VectorXf> to_vector(const std::vector<float>& coefficients)
{
    return Eigen::Map<const Eigen::VectorXf>(coefficients.data(), coefficients.size());
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* BLENDSHAPE_HPP_ */
