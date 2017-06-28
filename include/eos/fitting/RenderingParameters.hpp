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

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"

#include "eos/fitting/detail/optional_cerealisation.hpp"
#include "eos/fitting/detail/glm_cerealisation.hpp"
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"

#include "Eigen/Core"

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
 * In general, the convention is as follows:
 * 	 r_x = Pitch
 *	 r_y = Yaw. Positive means subject is looking left (we see her right cheek).
 *	 r_z = Roll. Positive means the subject's right eye is further down than the other one (he tilts his head to the right).
 * However, we're using a quaternion now to represent the rotation, and glm::eulerAngles() will give
 * slightly different angles (according to a different (undocumented)) convention. However, the
 * rotation is exactly the same! (i.e. they are represented by the same quaternion / rotation matrix).
 *
 * This should always represent all parameters necessary to render the model to an image, and be completely OpenGL compliant.
 */
class RenderingParameters
{
public:
	// Creates with default frustum...
	RenderingParameters() {};

	// Initialisation for Eigen::LevMarq
	// This creates the correct rotation quaternion in the case the angles were estimated/given by R*P*Y*v.
	// Angles given in radian.
	// Note: If you subsequently use RP::get_rotation() and glm::eulerAngles() on it, the angles you get out will be slightly different from the ones you put in here.
	// But they will describe the same rotation! Just in a different order. (i.e. the rotation matrix or quaternion for both of these two sets of angles is identical.)
	RenderingParameters(CameraType camera_type, Frustum camera_frustum, float r_x, float r_y, float r_z, float tx, float ty, int screen_width, int screen_height) : camera_type(camera_type), frustum(camera_frustum), t_x(tx), t_y(ty), screen_width(screen_width), screen_height(screen_height) {
		auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
		auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
		auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
		auto zxy_rotation_matrix = rot_mtx_z * rot_mtx_x * rot_mtx_y;
		rotation = glm::quat(zxy_rotation_matrix);
	};

	// This assumes estimate_sop was run on points with OpenCV viewport! I.e. y flipped.
	RenderingParameters(ScaledOrthoProjectionParameters ortho_params, int screen_width, int screen_height) : camera_type(CameraType::Orthographic), t_x(ortho_params.tx), t_y(ortho_params.ty), screen_width(screen_width), screen_height(screen_height) {
		rotation = glm::quat(ortho_params.R);
		const auto l = 0.0;
		const auto r = screen_width / ortho_params.s;
		const auto b = 0.0; // The b and t values are not tested for what happens if the SOP parameters
		const auto t = screen_height / ortho_params.s; // were estimated on a non-flipped viewport.
		frustum = Frustum(l, r, b, t);
	};

	auto get_camera_type() const {
		return camera_type;
	};

	glm::quat get_rotation() const {
		return rotation;
	};

	void set_rotation(glm::quat rotation_quaternion) {
		rotation = rotation_quaternion;
	};

	void set_translation(float t_x, float t_y) {
		this->t_x = t_x;
		this->t_y = t_y;
	};

	glm::mat4x4 get_modelview() const {
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

	Frustum get_frustum() const {
		return frustum;
	};

	void set_frustum(Frustum frustum) {
		this->frustum = frustum;
	};

	int get_screen_width() const {
		return screen_width;
	};

	void set_screen_width(int screen_width) {
		this->screen_width = screen_width;
	};

	int get_screen_height() const {
		return screen_height;
	};

	void set_screen_height(int screen_height) {
		this->screen_height = screen_height;
	};

private:
	CameraType camera_type = CameraType::Orthographic;
	Frustum frustum; // Can construct a glm::ortho or glm::perspective matrix from this.

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
		archive(CEREAL_NVP(camera_type), CEREAL_NVP(frustum), CEREAL_NVP(rotation), CEREAL_NVP(t_x), CEREAL_NVP(t_y), CEREAL_NVP(screen_width), CEREAL_NVP(screen_height));
	};
};

/**
 * Saves the rendering parameters for an image to a json file.
 *
 * @param[in] rendering_parameters An instance of class RenderingParameters.
 * @param[in] filename The file to write.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
inline void save_rendering_parameters(RenderingParameters rendering_parameters, std::string filename)
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
inline glm::vec4 get_opencv_viewport(int width, int height)
{
	return glm::vec4(0, height, width, -height);
};

/**
 * @brief Converts a glm::mat4x4 to an Eigen::Matrix<float, 4, 4>.
 *
 * Note: I think the last use of this function is below in
 * get_3x4_affine_camera_matrix().
 */
inline Eigen::Matrix<float, 4, 4> to_eigen(const glm::mat4x4& glm_matrix)
{
	// glm stores its matrices in col-major order in memory, Eigen too.
	//using MatrixXf4x4 = Eigen::Matrix<float, 4, 4>;
	//Eigen::Map<MatrixXf4x4> eigen_map(&glm_matrix[0][0]); // doesn't work, why do we get a const*?
	Eigen::Matrix<float, 4, 4> eigen_matrix;
	for (int r = 0; r < 4; ++r) {
		for (int c = 0; c < 4; ++c) {
			eigen_matrix(r, c) = glm_matrix[c][r]; // Not checked, but should be correct?
		}
	}
	return eigen_matrix;
};

/**
 * @brief Creates a 3x4 affine camera matrix from given fitting parameters. The
 * matrix transforms points directly from model-space to screen-space.
 *
 * This function is mainly used since the linear shape fitting fitting::fit_shape_to_landmarks_linear
 * expects one of these 3x4 affine camera matrices, as well as render::extract_texture.
 */
inline Eigen::Matrix<float, 3, 4> get_3x4_affine_camera_matrix(RenderingParameters params, int width, int height)
{
	const auto view_model = to_eigen(params.get_modelview());
	const auto ortho_projection = to_eigen(params.get_projection());
	using MatrixXf3x4 = Eigen::Matrix<float, 3, 4>;
	using Eigen::Matrix4f;
	const Matrix4f mvp = ortho_projection * view_model;

	glm::vec4 viewport(0, height, width, -height); // flips y, origin top-left, like in OpenCV
	// equivalent to what glm::project's viewport does, but we don't change z and w:
	Eigen::Matrix4f viewport_mat;
	viewport_mat << viewport[2] / 2.0f, 0.0f, 0.0f, viewport[2] / 2.0f + viewport[0],
                    0.0f,               viewport[3] / 2.0f, 0.0f, viewport[3] / 2.0f + viewport[1], 
                    0.0f,               0.0f,               1.0f, 0.0f,
                    0.0f,               0.0f,               0.0f, 1.0f;

	const Matrix4f full_projection_4x4 = viewport_mat * mvp;
	MatrixXf3x4 full_projection_3x4 = full_projection_4x4.block<3, 4>(0, 0); // we take the first 3 rows, but then set the last one to [0 0 0 1]
	// Use .block, possibly with the static template arguments!
	full_projection_3x4(2, 0) = 0.0f;
	full_projection_3x4(2, 1) = 0.0f;
	full_projection_3x4(2, 2) = 0.0f;
	full_projection_3x4(2, 3) = 1.0f;

	return full_projection_3x4;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* RENDERINGPARAMETERS_HPP_ */
