/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/linear_shape_fitting.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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

#ifndef LINEARSHAPEFITTING_HPP_
#define LINEARSHAPEFITTING_HPP_

#include "eos/morphablemodel/PcaModel.hpp"

#include "Eigen/Core"
#include "Eigen/QR"
#include "Eigen/Sparse"

#include <vector>
#include <cassert>
#include <optional>

namespace eos {
	namespace fitting {

/**
 * Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood solution of the shape coefficients) as proposed in [1].
 * It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).
 *
 * [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
 *
 * Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and may contain an error.
 * Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude). Why? Because we fit using the normalised basis?
 * Note: The standard deviations given should be a vector, i.e. different for each landmark. This is not implemented yet.
 *
 * @param[in] shape_model The Morphable Model whose shape (coefficients) are estimated.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from model to screen-space.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] base_face The base or reference face from where the fitting is started. Usually this would be the models mean face, which is what will be used if the parameter is not explicitly specified.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean).
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or std::nullopt to fit all coefficients.
 * @param[in] detector_standard_deviation The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
 * @param[in] model_standard_deviation The standard deviation of the 3D vertex points in the 3D model, projected to 2D (so the value is in pixels).
 * @return The estimated shape-coefficients (alphas).
 */
inline std::vector<float> fit_shape_to_landmarks_linear(const morphablemodel::PcaModel& shape_model, Eigen::Matrix<float, 3, 4> affine_camera_matrix, const std::vector<Eigen::Vector2f>& landmarks, const std::vector<int>& vertex_ids, Eigen::VectorXf base_face=Eigen::VectorXf(), float lambda=3.0f, std::optional<int> num_coefficients_to_fit= std::optional<int>(), std::optional<float> detector_standard_deviation= std::optional<float>(), std::optional<float> model_standard_deviation= std::optional<float>())
{
	assert(landmarks.size() == vertex_ids.size());

	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	const int num_coeffs_to_fit = num_coefficients_to_fit.value_or(shape_model.get_num_principal_components());
	const int num_landmarks = static_cast<int>(landmarks.size());

	if (base_face.size() == 0)
	{
		base_face = shape_model.get_mean();
	}

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	MatrixXf V_hat_h = MatrixXf::Zero(4 * num_landmarks, num_coeffs_to_fit);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		const auto& basis_rows = shape_model.get_rescaled_pca_basis_at_point(vertex_ids[i]); // In the paper, the orthonormal basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the rescaled basis is fine/better.
	        V_hat_h.block(row_index, 0, 3, V_hat_h.cols()) = basis_rows.block(0, 0, basis_rows.rows(), num_coeffs_to_fit);
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	
        // Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
        Eigen::SparseMatrix<float> P(3 * num_landmarks, 4 * num_landmarks);
        std::vector<Eigen::Triplet<float>> P_coefficients; // list of non-zeros coefficients
        for (int i = 0; i < num_landmarks; ++i) { // Note: could make this the inner-most loop.
            for (int x = 0; x < affine_camera_matrix.rows(); ++x) {
                for (int y = 0; y < affine_camera_matrix.cols(); ++y) {
                    P_coefficients.push_back(Eigen::Triplet<float>(3 * i + x, 4 * i + y, affine_camera_matrix(x, y)));
                }
            }
        }
        P.setFromTriplets(P_coefficients.begin(), P_coefficients.end());
	
        // The variances: Add the 2D and 3D standard deviations.
	// If the user doesn't provide them, we choose the following:
	// 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
	// 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
	// The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
	const float sigma_squared_2D = std::pow(detector_standard_deviation.value_or(std::sqrt(3.0f)), 2) + std::pow(model_standard_deviation.value_or(0.0f), 2);
	// We use a VectorXf, and later use .asDiagonal():
	const VectorXf Omega = VectorXf::Constant(3 * num_landmarks, 1.0f / sigma_squared_2D);
	// Earlier, we set Sigma in a for-loop and then computed Omega, but it was really unnecessary:
	// Sigma(i, i) = sqrt(sigma_squared_2D), but then Omega is Sigma.t() * Sigma (squares the diagonal) - so we just assign 1/sigma_squared_2D to Omega here.

	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	VectorXf y = VectorXf::Ones(3 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		y(3 * i) = landmarks[i][0];
		y((3 * i) + 1) = landmarks[i][1];
		//y((3 * i) + 2) = 1; // already 1, stays (homogeneous coordinate)
	}
	
        // The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	VectorXf v_bar = VectorXf::Ones(4 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		v_bar(4 * i) = base_face(vertex_ids[i] * 3);
		v_bar((4 * i) + 1) = base_face(vertex_ids[i] * 3 + 1);
		v_bar((4 * i) + 2) = base_face(vertex_ids[i] * 3 + 2);
		//v_bar((4 * i) + 3) = 1; // already 1, stays (homogeneous coordinate)
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega:
	const MatrixXf A = P * V_hat_h; // camera matrix times the basis
	const MatrixXf b = P * v_bar - y; // camera matrix times the mean, minus the landmarks
	const MatrixXf AtOmegaAReg = A.transpose() * Omega.asDiagonal() * A + lambda * Eigen::MatrixXf::Identity(num_coeffs_to_fit, num_coeffs_to_fit);
	const MatrixXf rhs = -A.transpose() * Omega.asDiagonal() * b; // It's -A^t*Omega^t*b, but we don't need to transpose Omega, since it's a diagonal matrix, and Omega^t = Omega.
	// c_s: The 'x' that we solve for. (The variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$.)
	// We get coefficients ~ N(0, 1), because we're fitting with the rescaled basis. The coefficients are not multiplied with their eigenvalues.
	const VectorXf c_s = AtOmegaAReg.colPivHouseholderQr().solve(rhs);

	return std::vector<float>(c_s.data(), c_s.data() + c_s.size());
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* LINEARSHAPEFITTING_HPP_ */
