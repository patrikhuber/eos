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

#include "eos/morphablemodel/io/mat_cerealisation.hpp"
#include "cereal/types/string.hpp"
#include "cereal/archives/binary.hpp"

#include "opencv2/core/core.hpp"

#include <string>
#include <vector>
#include <cassert>
#include <fstream>

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
	std::string name; ///< Name of the blendshape.
	cv::Mat deformation; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices. Has the same format as PcaModel::mean.

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(CEREAL_NVP(name), CEREAL_NVP(deformation));
	};
};

/**
 * Helper method to load a file with blendshapes from
 * a cereal::BinaryInputArchive from the harddisk.
 *
 * @param[in] filename Filename to a blendshapes-file.
 * @return The loaded blendshapes.
 * @throw std::runtime_error When the file given in \c filename fails to be opened (most likely because the file doesn't exist).
 */
std::vector<Blendshape> load_blendshapes(std::string filename)
{
	std::vector<Blendshape> blendshapes;

	std::ifstream file(filename, std::ios::binary);
	if (file.fail()) {
		throw std::runtime_error("Error opening given file: " + filename);
	}
	cereal::BinaryInputArchive input_archive(file);
	input_archive(blendshapes);

	return blendshapes;
};

/**
 * @brief Copies the blendshapes into a matrix, with each column being a blendshape.
 *
 * @param[in] blendshapes Vector of blendshapes.
 * @return The resulting matrix.
 */
cv::Mat to_matrix(const std::vector<Blendshape>& blendshapes)
{
	// assert: all blendshapes have to have the same number of rows, and one col
	assert(blendshapes.size() > 0);
	cv::Mat blendshapes_as_basis(blendshapes[0].deformation.rows, blendshapes.size(), CV_32FC1);
	for (int i = 0; i < blendshapes.size(); ++i)
	{
		blendshapes[i].deformation.copyTo(blendshapes_as_basis.col(i));
	}
	return blendshapes_as_basis;
};

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* BLENDSHAPE_HPP_ */
