/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/RenderingParameters.hpp
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

#ifndef RENDERINGPARAMETERS_HPP_
#define RENDERINGPARAMETERS_HPP_

#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/render/utils.hpp" // for to_mat()

//#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "eos/fitting/detail/optional_cerealisation.hpp"
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <string>
#include <fstream>
#include <stdexcept>

namespace eos {
	namespace fitting {

/**
 * @brief A class representing a camera viewing frustum. At the moment only
 * fully tested with orthographic camera.
 */
struct Frustum
{
//	float l, r, b, t;
	Frustum() {};
	Frustum(float l, float r, float b, float t) : l(l), r(r), b(b), t(t) {};
	//Frustum(float l, float r, float b, float t, float n, float f) : l(l), r(r), b(b), t(t), n(n), f(f) {};
	float l = -1.0f;
	float r = 1.0f;
	float b = -1.0f;
	float t = 1.0f;
	//boost::optional<float> n; // These are not needed yet but probably will in the future,
	//boost::optional<float> f; // and then it's good if the old serialised files stay compatible.

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(CEREAL_NVP(l), CEREAL_NVP(r), CEREAL_NVP(b), CEREAL_NVP(t));
	};
};

enum class CameraType
{
	Orthographic,
	Perspective
};

/**
 * @brief Represents a set of estimated model parameters (rotation, translation) and
 * camera parameters (viewing frustum).
 *
 * The estimated rotation and translation transform the model from model-space to camera-space,
 * and, if one wishes to use OpenGL, can be used to build the model-view matrix.
 * The parameters are the inverse of the camera position in 3D space.
 *
 * The camera frustum describes the size of the viewing plane of the camera, and
 * can be used to build an OpenGL-conformant orthographic projection matrix.
 *
 * Together, these parameters fully describe the imaging process of a given model instance
 * (under an orthographic projection).
 *
 * The rotation values are given in radians and estimated using the RPY convention.
 * Yaw is applied first to the model, then pitch, then roll (R * P * Y * vertex).
 *
 * This should always represent all parameters necessary to render the model to an image, and be completely OpenGL compliant.
 */
struct RenderingParameters
{
	// Creates with default frustum...
	RenderingParameters() {};

	// This assumes estimate_sop was run on points with OpenCV viewport! I.e. y flipped.
	RenderingParameters(ScaledOrthoProjectionParameters ortho_params, int image_width, int image_height) {
		camera_type = CameraType::Orthographic;
		rotation = ortho_params.R; // convert the rotation matrix to a quaternion
		t_x = ortho_params.tx;
		t_y = ortho_params.ty;
		const auto l = 0.0;
		const auto r = image_width / ortho_params.s;
		const auto b = 0.0; // The b and t values are not tested for what happens if the SOP parameters
		const auto t = image_height / ortho_params.s; // were estimated on a non-flipped viewport.
		frustum = Frustum(l, r, b, t);
	};

	glm::quat get_rotation() const {
		return rotation;
	};

	glm::mat4x4 get_modelview() const {
		// rot from quat, add transl., return 4x4.
		glm::mat4x4 modelview = glm::mat4_cast(rotation);
		modelview[3][0] = t_x;
		modelview[3][1] = t_y;
		return modelview;
	};
	
	glm::mat4x4 get_projection() const {
		if (camera_type == CameraType::Orthographic)
		{
			return glm::ortho<float>(frustum.l, frustum.r, frustum.b, frustum.t);
		}
		else {
			throw std::runtime_error("get_projection() for CameraType::Perspective is not implemented yet.");
		}
	};

	CameraType camera_type = CameraType::Orthographic;
	Frustum frustum; // Can construct a glm::ortho or glm::perspective matrix from this.
	
	// Todo: Get rid of the Euler angles and just use the quaternion.
	float r_x; // Pitch.
	float r_y; // Yaw. Positive means subject is looking left (we see her right cheek).
	float r_z; // Roll. Positive means the subject's right eye is further down than the other one (he tilts his head to the right).
	glm::quat rotation;
	
	float t_x;
	float t_y;
	//boost::optional<float> t_z;
	//boost::optional<float> focal_length; // only for certain camera types. Should it go into Frustum?

	int screen_width; // (why) do we need these?
	int screen_height;

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(CEREAL_NVP(camera_type), CEREAL_NVP(frustum), CEREAL_NVP(r_x), CEREAL_NVP(r_y), CEREAL_NVP(r_z), CEREAL_NVP(t_x), CEREAL_NVP(t_y), CEREAL_NVP(screen_width), CEREAL_NVP(screen_height));
	};
};

/**
 * Saves the rendering parameters for an image to a json file.
 *
 * @param[in] rendering_parameters An instance of class RenderingParameters.
 * @param[in] filename The file to write.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
void save_rendering_parameters(RenderingParameters rendering_parameters, std::string filename)
{
	std::ofstream file(filename);
	if (file.fail()) {
		throw std::runtime_error("Error opening file for writing: " + filename);
	}
	cereal::JSONOutputArchive output_archive(file);
	output_archive(cereal::make_nvp("rendering_parameters", rendering_parameters));
};

/**
 * @brief Returns a glm/OpenGL compatible viewport vector that flips y and
 * has the origin on the top-left, like in OpenCV.
 */
glm::vec4 get_opencv_viewport(int width, int height)
{
	return glm::vec4(0, height, width, -height);
};

/**
 * @brief Creates a 4x4 model-view matrix from given fitting parameters.
 *
 * Together with the Frustum information, this describes the full
 * orthographic rendering parameters of the OpenGL pipeline.
 * Example:
 *
 * @code
 * fitting::OrthographicRenderingParameters rendering_params = ...;
 * glm::mat4x4 view_model = get_4x4_modelview_matrix(rendering_params);
 * glm::mat4x4 ortho_projection = glm::ortho(rendering_params.frustum.l, rendering_params.frustum.r, rendering_params.frustum.b, rendering_params.frustum.t);
 * glm::vec4 viewport(0, image.rows, image.cols, -image.rows); // flips y, origin top-left, like in OpenCV
 *
 * // project a point from 3D to 2D:
 * glm::vec3 point_3d = ...; // from a mesh for example
 * glm::vec3 point_2d = glm::project(point_3d, view_model, ortho_projection, viewport);
 * @endcode
 */
glm::mat4x4 get_4x4_modelview_matrix(fitting::RenderingParameters params)
{
	// rotation order: RPY * P
	auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), params.r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
	auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), params.r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
	auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), params.r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
	auto t_mtx = glm::translate(glm::mat4(1.0f), glm::vec3{ params.t_x, params.t_y, 0.0f });
	auto modelview = t_mtx * rot_mtx_z * rot_mtx_x * rot_mtx_y;
	return modelview;
};

/**
 * @brief Creates a 3x4 affine camera matrix from given fitting parameters. The
 * matrix transforms points directly from model-space to screen-space.
 *
 * This function is mainly used since the linear shape fitting fitting::fit_shape_to_landmarks_linear
 * expects one of these 3x4 affine camera matrices, as well as render::extract_texture.
 */
cv::Mat get_3x4_affine_camera_matrix(fitting::RenderingParameters params, int width, int height)
{
	auto view_model = render::to_mat(get_4x4_modelview_matrix(params));
	auto ortho_projection = render::to_mat(glm::ortho(params.frustum.l, params.frustum.r, params.frustum.b, params.frustum.t));
	cv::Mat mvp = ortho_projection * view_model;

	glm::vec4 viewport(0, height, width, -height); // flips y, origin top-left, like in OpenCV
	// equivalent to what glm::project's viewport does, but we don't change z and w:
	cv::Mat viewport_mat = (cv::Mat_<float>(4, 4) << viewport[2] / 2.0f, 0.0f,       0.0f, viewport[2] / 2.0f + viewport[0],
												     0.0f,               viewport[3] / 2.0f, 0.0f, viewport[3] / 2.0f + viewport[1],
													 0.0f,               0.0f,               1.0f, 0.0f,
													 0.0f,               0.0f,               0.0f, 1.0f);

	cv::Mat full_projection_4x4 = viewport_mat * mvp;
	cv::Mat full_projection_3x4 = full_projection_4x4.rowRange(0, 3); // we take the first 3 rows, but then set the last one to [0 0 0 1]
	full_projection_3x4.at<float>(2, 0) = 0.0f;
	full_projection_3x4.at<float>(2, 1) = 0.0f;
	full_projection_3x4.at<float>(2, 2) = 0.0f;
	full_projection_3x4.at<float>(2, 3) = 1.0f;

	return full_projection_3x4;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* RENDERINGPARAMETERS_HPP_ */
