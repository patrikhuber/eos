/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/fitting.hpp
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

#ifndef FITTING_HPP_
#define FITTING_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/blendshape_fitting.hpp"

#include "opencv2/core/core.hpp"

#include <vector>

namespace eos {
	namespace fitting {

/**
 * Convenience function that fits the shape model and expression blendshapes to
 * landmarks. It iterates PCA-shape and blendshape fitting until convergence
 * (usually it converges within 5 to 10 iterations).
 *
 * Note/Todo: It would be great if the function gave access to the shape and
 * blendshape coefficients. Maybe add them as optional out parameters?
 *
 * @param[in] affine_camera_matrix The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] image_points 2D landmarks from an image to fit the model to.
 * @param[in] vertex_indices The vertex indices in the model that correspond to the 2D points.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @return The fitted model shape instance.
 */
cv::Mat fit_shape_model(cv::Mat affine_camera_matrix, eos::morphablemodel::MorphableModel morphable_model, std::vector<eos::morphablemodel::Blendshape> blendshapes, std::vector<cv::Vec2f> image_points, std::vector<int> vertex_indices, float lambda = 3.0f)
{
	using cv::Mat;
	
	Mat blendshapes_as_basis(blendshapes[0].deformation.rows, blendshapes.size(), CV_32FC1); // assert blendshapes.size() > 0 and all of them have same number of rows, and 1 col
	for (int i = 0; i < blendshapes.size(); ++i)
	{
		blendshapes[i].deformation.copyTo(blendshapes_as_basis.col(i));
	}

	std::vector<float> last_blendshape_coeffs, current_blendshape_coeffs; 
	std::vector<float> last_pca_coeffs, current_pca_coeffs;
	current_blendshape_coeffs.resize(blendshapes.size()); // starting values t_0, all zeros
	current_pca_coeffs.resize(morphable_model.get_shape_model().get_num_principal_components()); // starting values, all zeros
	Mat combined_shape;

	do // run at least once:
	{
		last_blendshape_coeffs = current_blendshape_coeffs;
		last_pca_coeffs = current_pca_coeffs;
		// Estimate the PCA shape coefficients with the current blendshape coefficients (0 in the first iteration):
		Mat mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() + blendshapes_as_basis * Mat(last_blendshape_coeffs);
		current_pca_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_camera_matrix, image_points, vertex_indices, mean_plus_blendshapes, lambda);

		// Estimate the blendshape coefficients with the current PCA model estimate:
		Mat pca_model_shape = morphable_model.get_shape_model().draw_sample(current_pca_coeffs);
		current_blendshape_coeffs = eos::fitting::fit_blendshapes_to_landmarks_linear(blendshapes, pca_model_shape, affine_camera_matrix, image_points, vertex_indices, 0.0f);

		combined_shape = pca_model_shape + blendshapes_as_basis * Mat(current_blendshape_coeffs);
	} while (std::abs(cv::norm(current_pca_coeffs) - cv::norm(last_pca_coeffs)) >= 0.01 || std::abs(cv::norm(current_blendshape_coeffs) - cv::norm(last_blendshape_coeffs)) >= 0.01);
	
	return combined_shape;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* FITTING_HPP_ */
