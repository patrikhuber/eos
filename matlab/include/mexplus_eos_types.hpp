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
 * @brief Converts a glm::tmat4x4<float> to a Matlab matrix.
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
 * @brief Converts an std::vector of glm::tvec3<float> to a Matlab matrix.
 *
 * This function converts a whole vector of vec3's to an n x 3 Matlab matrix,
 * where n is data.size(). It is mainly used to pass vertex colour data of
 * a Mesh to Matlab.
 *
 * See template<> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
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
 * See template<> mxArray* MxArray::from(const std::vector<glm::tvec2<float>>&)
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
 * @brief Define a template specialisation for eos::render::Mesh.
 *
 * This converts an eos::render::Mesh into a Matlab struct.
 * 
 * @param[in] mesh The Mesh that will be returned to Matlab.
 * @return An mxArray containing a Matlab struct with all vertex, colour, texcoords and triangle data.
 */
template<>
mxArray* MxArray::from(const eos::core::Mesh& mesh) {

	MxArray out_array(MxArray::Struct());
	out_array.set("vertices", mesh.vertices);
	out_array.set("colors", mesh.colors);
	out_array.set("texcoords", mesh.texcoords);
	out_array.set("tvi", mesh.tvi);
	out_array.set("tci", mesh.tci);

	return out_array.release();
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
