/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/blendshape_fitting.hpp
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

#ifndef BLENDSHAPEFITTING_HPP_
#define BLENDSHAPEFITTING_HPP_

#include "eos/morphablemodel/Blendshape.hpp"

#include "Eigen/Core"
#include "Eigen/QR"
#define EIGEN3_NNLS_DEBUG
#include "nnls.h"

#include "opencv2/core/core.hpp"

#include <vector>
#include <cassert>

namespace eos {
	namespace fitting {

/**
 * Fits blendshape coefficients to given 2D landmarks, given a current face shape instance.
 * It's a linear, closed-form solution fitting algorithm, with regularisation (constraining
 * the L2-norm of the coefficients). However, there is no constraint on the coefficients,
 * so negative coefficients are allowed, which, with linear blendshapes (offsets), will most
 * likely not be desired. Thus, prefer the function
 * fit_blendshapes_to_landmarks_nnls(std::vector<eos::morphablemodel::Blendshape>, cv::Mat, cv::Mat, std::vector<cv::Vec2f>, std::vector<int>).
 * 
 * This algorithm is very similar to the shape fitting in fit_shape_to_landmarks_linear.
 * Instead of the PCA basis, the blendshapes are used, and instead of the mean, a current
 * face instance is used to do the fitting from.
 *
 * @param[in] blendshapes A vector with blendshapes to estimate the coefficients for.
 * @param[in] face_instance A shape instance from which the blendshape coefficients should be estimated (i.e. the current mesh without expressions, e.g. estimated from a previous PCA-model fitting). A 3m x 1 matrix.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from model to screen-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image to fit the blendshapes to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] lambda A regularisation parameter, constraining the L2-norm of the coefficients.
 * @return The estimated blendshape-coefficients.
 */
inline std::vector<float> fit_blendshapes_to_landmarks_linear(const std::vector<eos::morphablemodel::Blendshape>& blendshapes, const Eigen::VectorXf& face_instance, cv::Mat affine_camera_matrix, const std::vector<cv::Vec2f>& landmarks, const std::vector<int>& vertex_ids, float lambda=500.0f)
{
	assert(landmarks.size() == vertex_ids.size());

	using cv::Mat;
	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	const int num_blendshapes = blendshapes.size();
	const int num_landmarks = static_cast<int>(landmarks.size());

	// Copy all blendshapes into a "basis" matrix with each blendshape being a column:
	MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	MatrixXf V_hat_h = MatrixXf::Zero(4 * num_landmarks, num_blendshapes);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		V_hat_h.block(row_index, 0, 3, V_hat_h.cols()) = blendshapes_as_basis.block(vertex_ids[i] * 3, 0, 3, blendshapes_as_basis.cols());
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	MatrixXf P = MatrixXf::Zero(3 * num_landmarks, 4 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		P.block(3 * i, 4 * i, 3, 4) = Eigen::Map<RowMajorMatrixXf>(affine_camera_matrix.ptr<float>(), affine_camera_matrix.rows, affine_camera_matrix.cols);
	}

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
		v_bar(4 * i) = face_instance(vertex_ids[i] * 3);
		v_bar((4 * i) + 1) = face_instance(vertex_ids[i] * 3 + 1);
		v_bar((4 * i) + 2) = face_instance(vertex_ids[i] * 3 + 2);
		//v_bar((4 * i) + 3) = 1; // already 1, stays (homogeneous coordinate)
	}

	// Bring into standard regularised quadratic form:
	const MatrixXf A = P * V_hat_h; // camera matrix times the basis
	const MatrixXf b = P * v_bar - y; // camera matrix times the mean, minus the landmarks
	
	const MatrixXf AtAReg = A.transpose() * A + lambda * Eigen::MatrixXf::Identity(num_blendshapes, num_blendshapes);
	const MatrixXf rhs = -A.transpose() * b;

	const VectorXf coefficients = AtAReg.colPivHouseholderQr().solve(rhs);

	return std::vector<float>(coefficients.data(), coefficients.data() + coefficients.size());
};

/**
 * Fits blendshape coefficients to given 2D landmarks, given a current face shape instance.
 * Uses non-negative least-squares (NNLS) to solve for the coefficients. The NNLS algorithm
 * used doesn't support any regularisation.
 *
 * This algorithm is very similar to the shape fitting in fit_shape_to_landmarks_linear.
 * Instead of the PCA basis, the blendshapes are used, and instead of the mean, a current
 * face instance is used to do the fitting from.
 *
 * @param[in] blendshapes A vector with blendshapes to estimate the coefficients for.
 * @param[in] face_instance A shape instance from which the blendshape coefficients should be estimated (i.e. the current mesh without expressions, e.g. estimated from a previous PCA-model fitting). A 3m x 1 matrix.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from model to screen-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image to fit the blendshapes to.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] landmarks_standard_deviation The standard deviation of the 2D landmarks given (e.g. of the detector used), in pixels.
 * @return The estimated blendshape-coefficients.
 */
inline std::vector<float> fit_blendshapes_to_landmarks_nnls(const std::vector<morphablemodel::Blendshape>& blendshapes, const Eigen::VectorXf& face_instance, cv::Mat affine_camera_matrix, const std::vector<cv::Vec2f>& landmarks, const std::vector<int>& vertex_ids, std::vector<float> landmarks_standard_deviation = std::vector<float>())
{
	assert(landmarks.size() == vertex_ids.size());
        assert(landmarks_standard_deviation.size() == landmarks.size() || landmarks_standard_deviation.empty());

	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	const int num_blendshapes = blendshapes.size();
	const int num_landmarks = static_cast<int>(landmarks.size());

	// Copy all blendshapes into a "basis" matrix with each blendshape being a column:
	const MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	MatrixXf V_hat_h = MatrixXf::Zero(4 * num_landmarks, num_blendshapes);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		V_hat_h.block(row_index, 0, 3, V_hat_h.cols()) = blendshapes_as_basis.block(vertex_ids[i] * 3, 0, 3, blendshapes_as_basis.cols());
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	MatrixXf P = MatrixXf::Zero(3 * num_landmarks, 4 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		P.block(3 * i, 4 * i, 3, 4) = Eigen::Map<RowMajorMatrixXf>(affine_camera_matrix.ptr<float>(), affine_camera_matrix.rows, affine_camera_matrix.cols);
	}
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	VectorXf y = VectorXf::Ones(3 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		y(3 * i) = landmarks[i][0];
		y((3 * i) + 1) = landmarks[i][1];
		//y_((3 * i) + 2) = 1; // already 1, stays (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	VectorXf v_bar = VectorXf::Ones(4 * num_landmarks);
	for (int i = 0; i < num_landmarks; ++i) {
		v_bar(4 * i) = face_instance(vertex_ids[i] * 3);
		v_bar((4 * i) + 1) = face_instance(vertex_ids[i] * 3 + 1);
		v_bar((4 * i) + 2) = face_instance(vertex_ids[i] * 3 + 2);
		//v_bar((4 * i) + 3) = 1; // already 1, stays (homogeneous coordinate)
	}

        // Set up what will be the diagonal distance matrix Omega:
        std::experimental::optional<float> model_standard_deviation; // empty for now, unused.
        VectorXf Omega;
        if (landmarks_standard_deviation.empty()) {
            //const float sigma_squared_2D = std::pow(std::sqrt(3.0f), 2) + std::pow(model_standard_deviation.value_or(0.0f), 2);
            //Omega.setConstant(3 * num_landmarks, 1.0f / sigma_squared_2D);
	    // If there are no standard deviations given, we leave Omega empty. If Omega is empty, Eigen::NNLS<>::solve will not use it.
        }
        else {
            Omega = VectorXf(3 * num_landmarks);
            const float model_sdev = std::pow(model_standard_deviation.value_or(0.0f), 2);
            for (int i = 0; i < num_landmarks; ++i) {
                Omega(3 * i + 0) = 1.0f / (std::pow(landmarks_standard_deviation[i], 2) + model_sdev);
                Omega(3 * i + 1) = 1.0f / (std::pow(landmarks_standard_deviation[i], 2) + model_sdev);
                Omega(3 * i + 2) = 1.0f / (std::pow(landmarks_standard_deviation[i], 2) + model_sdev);
            }
        }

	// Bring into standard least squares form:
	const MatrixXf A = P * V_hat_h; // camera matrix times the basis
	const MatrixXf b = P * v_bar - y; // camera matrix times the mean, minus the landmarks
	// Solve using NNLS:
	VectorXf coefficients;
	bool non_singular = Eigen::NNLS<MatrixXf>::solve(A, -b, coefficients, 50, 1e-10, Omega);

	return std::vector<float>(coefficients.data(), coefficients.data() + coefficients.size());
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* BLENDSHAPEFITTING_HPP_ */
