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
 * Todo: Add a doxygen file comment? See opencv cereal?
 */

namespace mexplus {

template<>
mxArray* MxArray::from(const glm::tquat<float>& quat)
{
	MxArray out_array(MxArray::Numeric<double>(1, 4));
	for (int c = 0; c < 4; ++c) {
		out_array.set(c, quat[c]);
	}
	return out_array.release();
};

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
 * @brief Define a template specialisation for ... .
 *
 * Todo: Documentation.
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
	out_array.set("camera_type", camera_type);
	out_array.set("rotation_quaternion", rendering_parameters.get_rotation());
	out_array.set("modelview", rendering_parameters.get_modelview());
	out_array.set("projection", rendering_parameters.get_projection());
	out_array.set("screen_width", rendering_parameters.get_screen_width());
	out_array.set("screen_height", rendering_parameters.get_screen_height());

	return out_array.release();
};

} /* namespace mexplus */

#endif /* MEXPLUS_EOS_TYPES_HPP_ */
