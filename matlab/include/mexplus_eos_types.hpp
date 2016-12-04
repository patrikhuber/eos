/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/include/mexplus_eos_types.hpp
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

#ifndef MEXPLUS_EOS_TYPES_HPP_
#define MEXPLUS_EOS_TYPES_HPP_

#include "eos/render/Mesh.hpp"

#include "mexplus/mxarray.h"

#include "glm/vec4.hpp"

#include "Eigen/Core"

#include "mex.h"

/**
 * Todo: Add a doxygen file comment? See opencv cereal?
 */

namespace mexplus {

// We have an overload for vector<tvec4<float>> directly because otherwise a cell
// will be used. However, such overload for tvec4<float> can be added without affecting
// this one, this overload takes precedence!
// Todo: Numeric<float> instead?
template<>
mxArray* MxArray::from(const std::vector<glm::tvec2<float>>& vec)
{
	MxArray out_array(MxArray::Numeric<double>(vec.size(), 2));
	for (int r = 0; r < vec.size(); ++r) {
		for (int c = 0; c < 2; ++c) {
			out_array.set(r, c, vec[r][c]);
		}
	}
	return out_array.release();
};

template<>
mxArray* MxArray::from(const std::vector<glm::tvec3<float>>& vec)
{
	MxArray out_array(MxArray::Numeric<double>(vec.size(), 3));
	for (int r = 0; r < vec.size(); ++r) {
		for (int c = 0; c < 3; ++c) {
			out_array.set(r, c, vec[r][c]);
		}
	}
	return out_array.release();
};

template<>
mxArray* MxArray::from(const std::vector<glm::tvec4<float>>& vec)
{
	MxArray out_array(MxArray::Numeric<double>(vec.size(), 4));
	for (int r = 0; r < vec.size(); ++r) {
		for (int c = 0; c < 4; ++c) {
			out_array.set(r, c, vec[r][c]);
		}
	}
	return out_array.release();
};

// This is for tvi and tci - return them as matrix, not as cell-array.
template<>
mxArray* MxArray::from(const std::vector<std::array<int, 3>>& vec)
{
	MxArray out_array(MxArray::Numeric<int>(vec.size(), 3));
	for (int r = 0; r < vec.size(); ++r) {
		for (int c = 0; c < 3; ++c) {
			out_array.set(r, c, vec[r][c]);
		}
	}
	return out_array.release();
};

/**
 * @brief Define a template specialisation for ... .
 *
 * Todo: Documentation.
 */
template<>
mxArray* MxArray::from(const eos::render::Mesh& mesh) {

	MxArray out_array(MxArray::Struct());
	out_array.set("vertices", mesh.vertices);
	out_array.set("colors", mesh.colors);
	out_array.set("texcoords", mesh.texcoords);
	out_array.set("tvi", mesh.tvi);
	out_array.set("tci", mesh.tci);

	return out_array.release();
};

/**
 * @brief Define a template specialisation for Eigen::MatrixXd for ... .
 *
 * Todo: Documentation.
 */
/*
template<>
void MxArray::to(const mxArray* in_array, Eigen::MatrixXd* eigen_matrix)
{
	MxArray array(in_array);

	if (array.dimensionSize() > 2)
	{
		mexErrMsgIdAndTxt("eos:matlab", "Given array has > 2 dimensions. Can only create 2-dimensional matrices (and vectors).");
	}

	if (array.dimensionSize() == 1 || array.dimensionSize() == 0)
	{
		mexErrMsgIdAndTxt("eos:matlab", "Given array has 0 or 1 dimensions but we expected a 2-dimensional matrix (or row/column vector).");
		// Even when given a single value dimensionSize() is 2, with n=m=1. When does this happen?
	}

	if (!array.isDouble())
	{
		mexErrMsgIdAndTxt("eos:matlab", "Trying to create a Eigen::MatrixXd in C++, but the given data is not of type double.");
	}

	// We can be sure now that the array is 2-dimensional (or 0, but then we're screwed anyway)
	auto nrows = array.dimensions()[0]; // or use array.rows()
	auto ncols = array.dimensions()[1];

	// I think I can just use Eigen::Matrix, not a Map - the Matrix c'tor that we call creates a Map anyway?
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigen_map(array.getData<double>(), nrows, ncols);
	// Not sure that's alright - who owns the data? I think as it is now, everything points to the data in the mxArray owned by Matlab, but I'm not 100% sure.
	*eigen_matrix = eigen_map;
};
*/
} /* namespace mexplus */

#endif /* MEXPLUS_EOS_TYPES_HPP_ */
