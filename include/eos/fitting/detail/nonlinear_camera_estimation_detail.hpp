/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp
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

#ifndef NONLINEARCAMERAESTIMATION_DETAIL_HPP_
#define NONLINEARCAMERAESTIMATION_DETAIL_HPP_

#include "glm/gtc/matrix_transform.hpp"

#include "Eigen/Geometry"

#include "opencv2/core/core.hpp"

#include <vector>

namespace eos {
	namespace fitting {
		namespace detail {


// ret: 3rd entry is the z
// radians
// expects the landmark points to be in opencv convention, i.e. origin TL
glm::vec3 project_ortho(glm::vec3 point, float rot_x_pitch, float rot_y_yaw, float rot_z_roll, float tx, float ty, float frustum_scale, /* fixed params now: */ int width, int height)
{
	// We could use quaternions in here, to be independent of the RPY... etc convention.
	// Then, the user can decompose the quaternion as he wishes to. But then we'd have to estimate 4 parameters?
	// This can of course be optimised, but we keep it this way while we're debugging and as long as it's not a performance issue.
	auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), rot_x_pitch, glm::vec3{ 1.0f, 0.0f, 0.0f });
	auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), rot_y_yaw, glm::vec3{ 0.0f, 1.0f, 0.0f });
	auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), rot_z_roll, glm::vec3{ 0.0f, 0.0f, 1.0f });
	auto t_mtx = glm::translate(glm::mat4(1.0f), glm::vec3{ tx, ty, 0.0f }); // glm: Col-major memory layout. [] gives the column

	// Note/Todo: Is this the full ortho? n/f missing? or do we need to multiply it with Proj...? See Shirley CG!
	// glm::frustum()?
	const float aspect = static_cast<float>(width) / height;
	auto ortho_mtx = glm::ortho(-1.0f * aspect * frustum_scale, 1.0f * aspect * frustum_scale, -1.0f * frustum_scale, 1.0f * frustum_scale);

	glm::vec4 viewport(0, height, width, -height); // flips y, origin top-left, like in OpenCV
	// P = RPY * P
	glm::vec3 res = glm::project(point, t_mtx * rot_mtx_z * rot_mtx_x * rot_mtx_y, ortho_mtx, viewport);
	return res;
};

/**
 * @brief Generic functor for Eigen's optimisation algorithms.
 */
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

	const int m_inputs, m_values;

	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }
};

/**
 * @brief LevenbergMarquardt cost function for the orthographic camera estimation.
 */
struct OrthographicParameterProjection : Functor<double>
{
public:
	// Creates a new OrthographicParameterProjection object with given data.
	OrthographicParameterProjection(std::vector<cv::Vec2f> image_points, std::vector<cv::Vec4f> model_points, int width, int height) : Functor<double>(6, image_points.size()), image_points(image_points), model_points(model_points), width(width), height(height) {};

	// x = current params, fvec = the errors/differences of the proj with current params and the GT (image_points)
	int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
	{
		const float aspect = static_cast<float>(width) / height;
		for (int i = 0; i < values(); i++)
		{
			// opencv to glm:
			glm::vec3 point_3d(model_points[i][0], model_points[i][1], model_points[i][2]);
			// projection given current params x:
			glm::vec3 proj_with_current_param_esti = project_ortho(point_3d, x[0], x[1], x[2], x[3], x[4], x[5], width, height);
			cv::Vec2f proj_point_2d(proj_with_current_param_esti.x, proj_with_current_param_esti.y);
			// diff of current proj to ground truth, our error
			auto diff = cv::norm(proj_point_2d, image_points[i]);
			// fvec should contain the differences
			// don't square it.
			fvec[i] = diff;
		}
		return 0;
	};

private:
	std::vector<cv::Vec2f> image_points;
	std::vector<cv::Vec4f> model_points;
	int width;
	int height;
};

		} /* namespace detail */
	} /* namespace fitting */
} /* namespace eos */

#endif /* NONLINEARCAMERAESTIMATION_DETAIL_HPP_ */
