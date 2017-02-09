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

#include "eos/core/Mesh.hpp"
#include "eos/fitting/RenderingParameters.hpp"

#include "mexplus/mxarray.h"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/mat4x4.hpp"

#include "Eigen/Core"

#include "mex.h"

#include <array>
#include <vector>

/**
 * @file
 * @brief Contains mexplus template specialisations to convert eos data
 * structures into Matlab.
 *
 * Note 1: These all copy the data, which I believe might be necessary, since
 * Matlab may unload a mex module (with all its allocated data) at any given
 * time.
 * Note 2: They all return double vectors and matrices, even when the data given
 * are floats. We can think about changing that if it's a speed issue, however,
 * I think double is Matlab's default data type.
 */

namespace mexplus {

/**
 * @brief Converts a glm::tquat<float> to a Matlab vector.
 *
 * @param[in] quat The quaternion to convert.
 * @return An 1x4 Matlab vector.
 */
template<>
mxArray* MxArray::from(const glm::tquat<float>& quat)
{
	MxArray out_array(MxArray::Numeric<double>(1, 4));
	for (int c = 0; c < 4; ++c) {
		out_array.set(c, quat[c]);
	}
	return out_array.release();
};

/**
 * @brief Converts a glm::tmat4x4<float> to a Matlab (double) matrix.
 *
 * @param[in] mat The matrix to convert.
 * @return A 4x4 Matlab matrix.
 */
template<>
mxArray* MxArray::from(const glm::tmat4x4<float>& mat)
{
	MxArray out_array(MxArray::Numeric<double>(4, 4));
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c) {
			out_array.set(r, c, mat[c][r]);
		}
	}
	return out_array.release();
};

/**
 * @brief Converts an std::vector of glm::tvec2<float> to a Matlab matrix.
 *
 * This function converts a whole vector of vec2's to an n x 2 Matlab matrix,
 * where n is data.size(). It is mainly used to pass texture coordinates of
 * a Mesh to Matlab.
 *
 * We specialise for std::vector<glm::tvec2<float>> directly (and not for
 * glm::tvec2<float>) because otherwise a cell array of vec2's would be
 * generated. Luckily, even if a tvec2 specialisation was to exist too,
 * this one would take precedence to convert a vector<tvec2>.
 *
 * @param[in] data The data to convert.
 * @return An n x 2 Matlab matrix.
 */
template<>
mxArray* MxArray::from(const std::vector<glm::tvec2<float>>& data)
{
	MxArray out_array(MxArray::Numeric<double>(data.size(), 2));
	for (int r = 0; r < data.size(); ++r) {
		for (int c = 0; c < 2; ++c) {
			out_array.set(r, c, data[r][c]);
		}
	}
	return out_array.release();
};

/**
 * @brief Converts a Matlab matrix into an std::vector of glm::vec2.
 *
 * This function converts an n x 2 Matlab matrix to a vector of vec2's.
 * It is mainly used to pass vertex texture coordinate data from Matlab back to C++.
 *
 * See template <> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
 * for more details regarding the type specialisation.
 *
 * @param[in] in_array The matrix data from Matlab.
 * @param[in,out] data The converted data in C++.
 */
template <>
void MxArray::to(const mxArray* in_array, std::vector<glm::vec2>* data)
{
    MxArray arr(in_array);

    if (arr.dimensionSize() != 2)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 2 dimensions, i.e. n x 2.");
    }
    if (arr.cols() != 2)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 2 elements in the second dimension, i.e. n x 2.");
    }
    if (!arr.isDouble())
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array should be of type double.");
    }

    const auto num_vertices = arr.rows();
    data->reserve(num_vertices);

    for (int i = 0; i < num_vertices; ++i)
    {
        data->push_back(
            glm::vec2(arr.at<double>(mwIndex(i), mwIndex(0)), arr.at<double>(mwIndex(i), mwIndex(1))));
    }
};

/**
 * @brief Converts an std::vector of glm::tvec3<float> to a Matlab matrix.
 *
 * This function converts a whole vector of vec3's to an n x 3 Matlab matrix,
 * where n is data.size(). It is mainly used to pass vertex colour data of
 * a Mesh to Matlab.
 *
 * See template <> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
 * for more details.
 *
 * @param[in] data The data to convert.
 * @return An n x 3 Matlab matrix.
 */
template<>
mxArray* MxArray::from(const std::vector<glm::tvec3<float>>& data)
{
	MxArray out_array(MxArray::Numeric<double>(data.size(), 3));
	for (int r = 0; r < data.size(); ++r) {
		for (int c = 0; c < 3; ++c) {
			out_array.set(r, c, data[r][c]);
		}
	}
	return out_array.release();
};

/**
 * @brief Converts an std::vector of glm::tvec4<float> to a Matlab matrix.
 *
 * This function converts a whole vector of vec4's to an n x 4 Matlab matrix,
 * where n is data.size(). It is mainly used to pass vertex data of a Mesh
 * to Matlab.
 *
 * See template <> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
 * for more details.
 *
 * @param[in] data The data to convert.
 * @return An n x 4 Matlab matrix.
 */
template<>
mxArray* MxArray::from(const std::vector<glm::tvec4<float>>& data)
{
	MxArray out_array(MxArray::Numeric<double>(data.size(), 4));
	for (int r = 0; r < data.size(); ++r) {
		for (int c = 0; c < 4; ++c) {
			out_array.set(r, c, data[r][c]);
		}
	}
	return out_array.release();
};

/**
 * @brief Converts a Matlab matrix into an std::vector of glm::vec4.
 *
 * This function converts an n x 4 Matlab matrix to a vector of vec4's.
 * It is mainly used to pass vertex data from Matlab back to C++.
 *
 * See template <> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
 * for more details regarding the type specialisation.
 *
 * @param[in] in_array The matrix data from Matlab.
 * @param[in,out] data The converted data in C++.
 */
template <>
void MxArray::to(const mxArray* in_array, std::vector<glm::vec4>* data)
{
    MxArray arr(in_array);

    if (arr.dimensionSize() != 2)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 2 dimensions, i.e. n x 4.");
    }
    if (arr.cols() != 4)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 4 elements in the second dimension, i.e. n x 4.");
    }
    if (!arr.isDouble())
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array should be of type double.");
    }

    const auto num_vertices = arr.rows();
    data->reserve(num_vertices);

    for (int i = 0; i < num_vertices; ++i)
    {
        data->push_back(
            glm::vec4(arr.at<double>(mwIndex(i), mwIndex(0)), arr.at<double>(mwIndex(i), mwIndex(1)),
                      arr.at<double>(mwIndex(i), mwIndex(2)), arr.at<double>(mwIndex(i), mwIndex(3))));
    }
};

/**
 * @brief Converts an std::vector of std::array<int, 3> to a Matlab matrix.
 *
 * This function converts a whole vector of array<int, 3>'s to an n x 3 Matlab
 * matrix, where n is data.size(). It is mainly used to pass triangle indices
 * data of a Mesh to Matlab.
 *
 * We specialise for vector<array<int, 3>> directly (and not for array<int, 3>)
 * because otherwise a cell array of arrays would be generated. Luckily, even
 * if an array<int, 3> specialisation was to exist too, this one would take
 * precedence to convert a vector<array<int, 3>>.
 *
 * @param[in] data The data to convert.
 * @return An n x 3 Matlab matrix.
 */
template<>
mxArray* MxArray::from(const std::vector<std::array<int, 3>>& data)
{
	MxArray out_array(MxArray::Numeric<int>(data.size(), 3));
	for (int r = 0; r < data.size(); ++r) {
		for (int c = 0; c < 3; ++c) {
			out_array.set(r, c, data[r][c]);
		}
	}
	return out_array.release();
};

/**
 * @brief Converts a Matlab matrix into an std::vector of std::array<int, 3>.
 *
 * This function converts a n x 3 Matlab matrix to a vector<array<int, 3>>.
 * It is mainly used to pass triangle indices data of a Mesh to Matlab.
 *
 * See template <> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
 * for more details regarding the type specialisation.
 *
 * @param[in] in_array The matrix data from Matlab.
 * @param[in,out] data The converted data in C++.
 */
template <>
void MxArray::to(const mxArray* in_array, std::vector<std::array<int, 3>>* data)
{
    MxArray arr(in_array);

    if (arr.dimensionSize() != 2)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 2 dimensions, i.e. n x 3.");
    }
    if (arr.cols() != 3)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array has to have 3 elements in the second dimension, i.e. n x 3.");
    }
    if (!arr.isInt32())
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given array should be of type int32.");
    }

    const auto num_faces = arr.rows();
    data->reserve(num_faces);

    for (int i = 0; i < num_faces; ++i)
    {
        data->push_back(std::array<int, 3>{arr.at<std::int32_t>(mwIndex(i), mwIndex(0)),
                                           arr.at<std::int32_t>(mwIndex(i), mwIndex(1)),
                                           arr.at<std::int32_t>(mwIndex(i), mwIndex(2))});
    }
};

/**
 * @brief Convert an eos::core::Mesh into a Matlab struct.
 *
 * Adjusts the triangle indices from 0-based (C++) to 1-based (Matlab).
 *
 * @param[in] mesh The Mesh that will be returned to Matlab.
 * @return An mxArray containing a Matlab struct with all vertex, colour, texcoords and triangle data.
 */
template<>
mxArray* MxArray::from(const eos::core::Mesh& mesh)
{
	// C++ counts the vertex indices starting at zero, Matlab starts counting
	// at one - therefore, add +1 to all triangle indices:
	auto tvi_1based = mesh.tvi;
	for (auto&& t : tvi_1based) {
		for (auto&& idx : t) {
			idx += 1;
		}
	}
	// Same for tci:
	auto tci_1based = mesh.tci;
	for (auto&& t : tci_1based) {
		for (auto&& idx : t) {
			idx += 1;
		}
	}

	MxArray out_array(MxArray::Struct());
	out_array.set("vertices", mesh.vertices);
	out_array.set("colors", mesh.colors);
	out_array.set("texcoords", mesh.texcoords);
	out_array.set("tvi", tvi_1based);
	out_array.set("tci", tci_1based);

	return out_array.release();
};

/**
 * @brief Convert a Matlab mesh struct back into an eos::core::Mesh.
 *
 * Adjusts the triangle indices from 1-based (Matlab) to 0-based (C++).
 *
 * @param[in] in_array Input mesh data from Matlab.
 * @param[in,out] mesh Converted eos::core::Mesh.
 */
template <>
void MxArray::to(const mxArray* in_array, eos::core::Mesh* mesh)
{
    MxArray array(in_array);

    if (!array.isStruct())
    {
        mexErrMsgIdAndTxt("eos:matlab", "Given mesh is not a Matlab struct.");
    }

    if (!array.hasField("vertices") || !array.hasField("tvi"))
    {
        mexErrMsgIdAndTxt("eos:matlab",
                          "Given mesh struct must contain at least the fields 'vertices' and 'tvi'.");
    }

    // We could check whether num_vertices is equal for these, but we'll leave it up to the user to give us
    // valid mesh data.
    array.at("vertices", &mesh->vertices);   // num_vertices x 4 double
    array.at("texcoords", &mesh->texcoords); // num_vertices x 2 double
    array.at("tvi", &mesh->tvi);             // num_faces x 3 int32

    // Adjust the vertex indices from 1-based (Matlab) to 0-based (C++):
    for (auto&& t : mesh->tvi)
    {
        t[0] -= 1;
        t[1] -= 1;
        t[2] -= 1;
    }
};

/**
 * @brief Define a template specialisation for eos::fitting::RenderingParameters.
 *
 * This converts an eos::fitting::RenderingParameters into a Matlab struct.
 * 
 * @param[in] rendering_parameters The RenderingParameters that will be returned to Matlab.
 * @return An mxArray containing a Matlab struct with all required parameters.
 */
template<>
mxArray* MxArray::from(const eos::fitting::RenderingParameters& rendering_parameters) {

	MxArray out_array(MxArray::Struct());
	
	const std::string camera_type = [&rendering_parameters]() {
		if (rendering_parameters.get_camera_type() == eos::fitting::CameraType::Orthographic)
		{
			return "Orthographic";
		}
		else if (rendering_parameters.get_camera_type() == eos::fitting::CameraType::Perspective) {
			return "Perspective";
		}
		else {
			return "unknown";
		}
	}();

	// Since we don't expose get_opencv_viewport(), and Matlab doesn't have glm::project()
	// anyway, we'll make a 4x4 viewport matrix available. Matlab seems to have the same
	// convention as OpenCV (top-left is the image origin).
	auto viewport = eos::fitting::get_opencv_viewport(rendering_parameters.get_screen_width(), rendering_parameters.get_screen_height());
	glm::mat4x4 viewport_matrix; // Identity matrix
	viewport_matrix[0][0] = 0.5f * viewport[2];
	viewport_matrix[3][0] = 0.5f * viewport[2] + viewport[0];
	viewport_matrix[1][1] = 0.5f * viewport[3];
	viewport_matrix[3][1] = 0.5f * viewport[3] + viewport[1];
	viewport_matrix[2][2] = 0.5f;
	viewport_matrix[3][2] = 0.5f;

	out_array.set("camera_type", camera_type);
	out_array.set("rotation_quaternion", rendering_parameters.get_rotation());
	out_array.set("modelview", rendering_parameters.get_modelview());
	out_array.set("projection", rendering_parameters.get_projection());
	out_array.set("viewport", viewport_matrix);
	out_array.set("screen_width", rendering_parameters.get_screen_width());
	out_array.set("screen_height", rendering_parameters.get_screen_height());

	return out_array.release();
};

} /* namespace mexplus */

#endif /* MEXPLUS_EOS_TYPES_HPP_ */
