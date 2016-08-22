/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/ceres_nonlinear.hpp
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

#ifndef CERESNONLINEAR_HPP_
#define CERESNONLINEAR_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"

#include "glm/gtc/quaternion.hpp"

#include "opencv2/core/core.hpp" // for Vec2f

#include <array>
#include <vector>

namespace eos {
	namespace fitting {

// Forward declarations:
template <typename T>
std::array<T, 3> get_shape_point(const morphablemodel::PcaModel& shape_model, const std::vector<morphablemodel::Blendshape>& blendshapes, int vertex_id, const T* const shape_coeffs, const T* const blendshape_coeffs);

template <typename T>
std::array<T, 3> get_vertex_colour(const morphablemodel::PcaModel& colour_model, int vertex_id, const T* const colour_coeffs);

/**
 * Cost function for a prior on the parameters.
 * 
 * Prior towards zero (0, 0...) for the parameters.
 * Note: The weight is inside the norm, so may not correspond to the "usual"
 * formulas. However I think it's equivalent up to a scaling factor, but it
 * should be checked.
 */
struct PriorCost {

	/**
	 * Creates a new prior object with set number of variables and a weight.
	 *
	 * @param[in] num_variables Number of variables that the parameter vector contains.
	 * @param[in] weight A weight that the parameters are multiplied with.
	 */
	PriorCost(int num_variables, double weight = 1.0) : num_variables(num_variables), weight(weight) {};

	/**
	 * Cost function implementation.
	 *
	 * @param[in] x An array of parameters.
	 * @param[in] residual An array of the resulting residuals.
	 * @return Returns true. The ceres documentation is not clear about that I think.
	 */
	template <typename T>
	bool operator()(const T* const x, T* residual) const {
		for (int i=0; i < num_variables; ++i)
		{
			residual[i] = weight * x[i];
		}
		return true;
	};

private:
	int num_variables;
	double weight;
};

/**
 * 2D landmark error cost function.
 * 
 * Computes the landmark reprojection error in 2D.
 * Models the cost for one landmark. The residual is 2-dim, [x, y].
 * Its input params are camera parameters, shape coefficients and
 * blendshape coefficients.
 */
struct LandmarkCost {

	/**
	 * Constructs a new landmark cost function object with for a particular landmark/vertex id.
	 *
	 * Warning: Don't put in temporaries for \c shape_model and \c blendshapes! We don't make a copy, we store a reference to what is given to the function!
	 *
	 * @param[in] shape_model A PCA 3D shape model. Do not use a temporary.
	 * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
	 * @param[in] observed_landmark An observed 2D landmark in an image.
	 * @param[in] vertex_id The vertex id that the given observed landmark corresponds to.
	 * @param[in] image_width Width of the image that the 2D landmark is from (needed for the model projection).
	 * @param[in] image_height Height of the image.
	 * @param[in] use_perspective Whether a perspective or an orthographic projection should be used.
	 */
	LandmarkCost(const morphablemodel::PcaModel& shape_model, const std::vector<morphablemodel::Blendshape>& blendshapes, cv::Vec2f observed_landmark, int vertex_id, int image_width, int image_height, bool use_perspective) : shape_model(shape_model), blendshapes(blendshapes), observed_landmark(observed_landmark), vertex_id(vertex_id), image_width(image_width), image_height(image_height), aspect_ratio(static_cast<double>(image_width) / image_height), use_perspective(use_perspective) {};

	/**
	 * Landmark cost function implementation.
	 *
	 * Measures the landmark reprojection error of the model with the estimated parameters and the observed 2D landmarks.
	 * For one single landmark.
	 *
	 * @param[in] camera_rotation A set of camera parameters, parameterised as a quaternion [w x y z].
	 * @param[in] camera_translation_and_intrinsics Camera translation and intrinsic parameters. Ortho: [t_x t_y frustum_scale]. Perspective: [t_x t_y t_z fov].
	 * @param[in] shape_coeffs A set of PCA shape coefficients.
	 * @param[in] blendshape_coeffs A set of blendshape coefficients.
	 * @param[in] residual An array of the resulting residuals.
	 * @return Returns true. The ceres documentation is not clear about that I think.
	 */
	template <typename T>
	bool operator()(const T* const camera_rotation, const T* const camera_translation_and_intrinsics, const T* const shape_coeffs, const T* const blendshape_coeffs, T* residual) const
	{
		using namespace glm;
		// Generate shape instance (of only one vertex id!) using current parameters and 10 shape coefficients:
		// Note: Why are we not returning a glm::tvec3<T>?
		const auto point_arr = get_shape_point<T>(shape_model, blendshapes, vertex_id, shape_coeffs, blendshape_coeffs);

		// Project the point to 2D:
		const tvec3<T> point_3d(point_arr[0], point_arr[1], point_arr[2]);
		// I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
		const tquat<T> rot_quat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
		// We rotate ZXY*p, which is RPY*p. I'm not sure this matrix still corresponds to RPY - probably if we use glm::eulerAngles(), these are not RPY anymore and we'd have to adjust if we were to use rotation matrices.
		const tmat4x4<T> rot_mtx = glm::mat4_cast(rot_quat);
		
		// Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
		const tvec4<T> viewport(0, image_height, image_width, -image_height); // OpenCV convention

		tvec3<T> projected_point; // Note: could avoid default construction by using a lambda and immediate invocation
		if (use_perspective)
		{
			const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], camera_translation_and_intrinsics[2]));
			const T& fov = camera_translation_and_intrinsics[3];
			const auto persp_mtx = glm::perspective(fov, T(aspect_ratio), T(0.1), T(1000.0));
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, persp_mtx, viewport);
		}
		else {
			const T& frustum_scale = camera_translation_and_intrinsics[2];
			const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], 0.0)); // we don't have t_z in ortho camera, it doesn't matter where it is
			const auto ortho_mtx = glm::ortho(-1.0 * aspect_ratio * frustum_scale, 1.0 * aspect_ratio * frustum_scale, -1.0 * frustum_scale, 1.0 * frustum_scale);
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, ortho_mtx, viewport);
		}
		// Residual: Projected point minus the observed 2D landmark point
		residual[0] = projected_point.x - T(observed_landmark[0]);
		residual[1] = projected_point.y - T(observed_landmark[1]);
		return true;
	};

private:
	const morphablemodel::PcaModel& shape_model; // Or store as pointer (non-owning) or std::reference_wrapper?
	const std::vector<morphablemodel::Blendshape>& blendshapes;
	const cv::Vec2f observed_landmark;
	const int vertex_id;
	const int image_width;
	const int image_height;
	const double aspect_ratio;
	const bool use_perspective;
};

/**
 * Image error cost function (at vertex locations).
 * 
 * Measures the RGB image error between a particular vertex point of the 3D
 * model at its projected location and the observed input image.
 * Models the cost for 1 vertex. The residual is 3-dim, [r, g, b].
 * Its input params are cam, shape-coeffs, BS-coeffs and colour coeffs.
 * This projects the vertex locations - so not a full rendering pass.
 */
struct ImageCost {

	/**
	 * Constructs a new cost function object for a particular vertex id that measures the RGB image error between the estimated model point and the observed input image.
	 *
	 * Warning: Don't put in temporaries for \c morphable_model and \c blendshapes! We don't make a copy, we store a reference to what is given to the function!
	 *
	 * @param[in] morphable_model A 3D Morphable Model. Do not use a temporary.
	 * @param[in] blendshapes A set of 3D blendshapes. Do not use a temporary.
	 * @param[in] image The observed image. TODO: We should assert that the image we get is 8UC3!
	 * @param[in] vertex_id Vertex id of the 3D model that should be projected and measured.
	 * @param[in] use_perspective Whether a perspective or an orthographic projection should be used.
	 * @throws std::runtime_error if the given \c image is not of type CV_8UC3.
	 */
	ImageCost(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, cv::Mat image, int vertex_id, bool use_perspective) : morphable_model(morphable_model), blendshapes(blendshapes), image(image), aspect_ratio(static_cast<double>(image.cols) / image.rows), vertex_id(vertex_id), use_perspective(use_perspective)
	{
		if (image.type() != CV_8UC3)
		{
			throw std::runtime_error("The image given to ImageCost must be of type CV_8UC3.");
		}
	};

	/**
	 * Image cost function implementation.
	 *
	 * Measures the image pixel error between the given model vertex projected to 2D and the observed input image.
	 *
	 * Todo: We should deal with visibility! Don't evaluate when the vertex is self-occluded.
	 *
	 * @param[in] camera_rotation A set of camera parameters, parameterised as a quaternion [w x y z].
	 * @param[in] camera_translation_and_intrinsics Camera translation and intrinsic parameters. Ortho: [t_x t_y frustum_scale]. Perspective: [t_x t_y t_z fov].
	 * @param[in] shape_coeffs A set of PCA shape coefficients.
	 * @param[in] blendshape_coeffs A set of blendshape coefficients.
	 * @param[in] color_coeffs A set of PCA colour (albedo) coefficients.
	 * @param[in] residual An array of the resulting residuals.
	 * @return Returns true. The ceres documentation is not clear about that I think.
	 */
	template <typename T>
	bool operator()(const T* const camera_rotation, const T* const camera_translation_and_intrinsics, const T* const shape_coeffs, const T* const blendshape_coeffs, const T* const color_coeffs, T* residual) const
	{
		using namespace glm;
		// Note: The following is all duplicated code with LandmarkCost. Fix if possible performance-wise.
		// Generate 3D shape point using the current parameters:
		const auto point_arr = get_shape_point<T>(morphable_model.get_shape_model(), blendshapes, vertex_id, shape_coeffs, blendshape_coeffs);

		// Project the point to 2D:
		const tvec3<T> point_3d(point_arr[0], point_arr[1], point_arr[2]);
		// I think the quaternion is always normalised because we run Ceres with QuaternionParameterization
		const tquat<T> rot_quat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
		// We rotate ZXY*p, which is RPY*p. I'm not sure this matrix still corresponds to RPY - probably if we use glm::eulerAngles(), these are not RPY anymore and we'd have to adjust if we were to use rotation matrices.
		const tmat4x4<T> rot_mtx = glm::mat4_cast(rot_quat);

		// Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
		const tvec4<T> viewport(0, image.rows, image.cols, -image.rows); // OpenCV convention

		tvec3<T> projected_point;
		if (use_perspective)
		{
			const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], camera_translation_and_intrinsics[2]));
			const T& focal = camera_translation_and_intrinsics[3];
			const auto persp_mtx = glm::perspective(focal, T(aspect_ratio), T(0.1), T(1000.0));
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, persp_mtx, viewport);
		}
		else {
			const T& frustum_scale = camera_translation_and_intrinsics[2];
			const auto t_mtx = glm::translate(tvec3<T>(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], 0.0)); // we don't have t_z in ortho camera, it doesn't matter where it is
			const auto ortho_mtx = glm::ortho(-1.0 * aspect_ratio * frustum_scale, 1.0 * aspect_ratio * frustum_scale, -1.0 * frustum_scale, 1.0 * frustum_scale);
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, ortho_mtx, viewport);
		}

		// Access the image colour value at the projected pixel location, if inside the image - otherwise set to (127, 127, 127) (maybe not ideal...):
		if (projected_point.y < T(0) || projected_point.y >= T(image.rows) || projected_point.x < T(0) || projected_point.x >= T(image.cols))
		{
			// The point is outside the image.
			residual[0] = T(127.0);
			residual[1] = T(127.0);
			residual[2] = T(127.0);

			// TODO: What does interpolator.Evaluate() return in this case?
		/*	Grid2D<uchar, 3> grid(image.ptr(0), 0, image.rows, 0, image.cols);
			BiCubicInterpolator<Grid2D<uchar, 3>> interpolator(grid);
			T observed_colour[3];
			interpolator.Evaluate(projected_y, projected_x, &observed_colour[0]); // says it returns false when (r, c) is out of bounds... but it returns void?
			//std::cout << observed_colour[0] << ", " << observed_colour[1] << ", " << observed_colour[2] << "\n";
			*/
			// It kind of looks like as if when it's out of bounds, there will be a vector out of bound access and an assert/crash?
			// No, in debugging, it looks like it just interpolates or something. Not clear currently.

		}
		else {
			// Note: We could store the BiCubicInterpolator as member variable.
			// The default template arguments for Grid2D are <T, kDataDim=1, kRowMajor=true, kInterleaved=true> and (except for the dimension), they're the right ones for us.
			ceres::Grid2D<uchar, 3> grid(image.ptr(0), 0, image.rows, 0, image.cols);
			ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> interpolator(grid);
			T observed_colour[3];
			interpolator.Evaluate(projected_point.y, projected_point.x, &observed_colour[0]);

			// This probably needs to be modified if we add a light model.
			auto model_colour = get_vertex_colour(morphable_model.get_color_model(), vertex_id, color_coeffs);
			// I think this returns RGB, and between [0, 1].

			// Residual: Vertex colour of model point minus the observed colour in the 2D image
			// observed_colour is BGR, model_colour is RGB. Residual will be RGB.
			residual[0] = model_colour[0] * 255.0 - T(observed_colour[2]);
			residual[1] = model_colour[1] * 255.0 - T(observed_colour[1]);
			residual[2] = model_colour[2] * 255.0 - T(observed_colour[0]);
		}
		return true;
	};

private:
	const morphablemodel::MorphableModel& morphable_model; // Or store as pointer (non-owning) or std::reference_wrapper?
	const std::vector<morphablemodel::Blendshape>& blendshapes;
	const cv::Mat image; // the observed image
	const double aspect_ratio;
	const int vertex_id;
	const bool use_perspective;
};

/**
 * Returns the 3D position of a single point of the 3D shape generated by the parameters given.
 *
 * @param[in] shape_model A PCA 3D shape model.
 * @param[in] blendshapes A set of 3D blendshapes.
 * @param[in] vertex_id Vertex id of the 3D model that should be projected.
 * @param[in] shape_coeffs A set of PCA shape coefficients used to generate the point.
 * @param[in] blendshape_coeffs A set of blendshape coefficients used to generate the point.
 * @return The 3D point.
 */
template <typename T>
std::array<T, 3> get_shape_point(const morphablemodel::PcaModel& shape_model, const std::vector<morphablemodel::Blendshape>& blendshapes, int vertex_id, const T* const shape_coeffs, const T* const blendshape_coeffs)
{
	int num_coeffs_fitting = 10; // Todo: Should be inferred or a function parameter!
	auto mean = shape_model.get_mean_at_point(vertex_id);
	auto basis = shape_model.get_normalised_pca_basis(vertex_id);
	// Computing Shape = mean + basis * coeffs:
	// Note: Could use an Eigen matrix with type T to see if it gives a speedup.
	std::array<T, 3> point{ T(mean[0]), T(mean[1]), T(mean[2]) };
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[0] += T(basis.row(0).col(i).at<float>(0)) * shape_coeffs[i]; // it seems to be ~15% faster when these are static_cast<double>() instead of T()?
	}
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[1] += T(basis.row(1).col(i).at<float>(0)) * shape_coeffs[i];
	}
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[2] += T(basis.row(2).col(i).at<float>(0)) * shape_coeffs[i];
	}
	// Adding the blendshape offsets: 
	// Shape = mean + basis * coeffs + blendshapes * bs_coeffs:
	auto num_blendshapes = blendshapes.size();
	for (int i = 0; i < num_blendshapes; ++i) {
		point[0] += T(blendshapes[i].deformation.at<float>(3 * vertex_id + 0)) * blendshape_coeffs[i];
	}
	for (int i = 0; i < num_blendshapes; ++i) {
		point[1] += T(blendshapes[i].deformation.at<float>(3 * vertex_id + 1)) * blendshape_coeffs[i];
	}
	for (int i = 0; i < num_blendshapes; ++i) {
		point[2] += T(blendshapes[i].deformation.at<float>(3 * vertex_id + 2)) * blendshape_coeffs[i];
	}
	return point;
};

/**
 * Returns the colour value of a single point of the 3D model generated by the parameters given.
 *
 * @param[in] color_model A PCA 3D colour (albedo) model.
 * @param[in] vertex_id Vertex id of the 3D model whose colour is to be returned.
 * @param[in] color_coeffs A set of PCA colour coefficients.
 * @return The colour. As RGB? In [0, 1]?
 */
template <typename T>
std::array<T, 3> get_vertex_colour(const morphablemodel::PcaModel& color_model, int vertex_id, const T* const color_coeffs)
{
	int num_coeffs_fitting = 10; // Todo: Should be inferred or a function parameter!
	auto mean = color_model.get_mean_at_point(vertex_id);
	auto basis = color_model.get_normalised_pca_basis(vertex_id);
	// Computing Colour = mean + basis * coeffs
	// Note: Could use an Eigen matrix with type T to see if it gives a speedup.
	std::array<T, 3> point{ T(mean[0]), T(mean[1]), T(mean[2]) };
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[0] += T(basis.row(0).col(i).at<float>(0)) * color_coeffs[i]; // it seems to be ~15% faster when these are static_cast<double>() instead of T()?
	}
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[1] += T(basis.row(1).col(i).at<float>(0)) * color_coeffs[i];
	}
	for (int i = 0; i < num_coeffs_fitting; ++i) {
		point[2] += T(basis.row(2).col(i).at<float>(0)) * color_coeffs[i];
	}
	return point;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* CERESNONLINEAR_HPP_ */
