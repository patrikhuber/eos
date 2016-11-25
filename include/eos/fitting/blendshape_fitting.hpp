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
inline std::vector<float> fit_blendshapes_to_landmarks_linear(std::vector<eos::morphablemodel::Blendshape> blendshapes, cv::Mat face_instance, cv::Mat affine_camera_matrix, std::vector<cv::Vec2f> landmarks, std::vector<int> vertex_ids, float lambda=500.0f)
{
	using cv::Mat;
	assert(landmarks.size() == vertex_ids.size());

	int num_coeffs_to_fit = blendshapes.size();
	int num_landmarks = static_cast<int>(landmarks.size());

	// Copy all blendshapes into a "basis" matrix with each blendshape being a column:
	cv::Mat blendshapes_as_basis(blendshapes[0].deformation.rows, blendshapes.size(), CV_32FC1); // assert blendshapes.size() > 0 and all of them have same number of rows, and 1 col
	for (int i = 0; i < blendshapes.size(); ++i)
	{
		blendshapes[i].deformation.copyTo(blendshapes_as_basis.col(i));
	}

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		//Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
		Mat basis_rows = blendshapes_as_basis.rowRange(vertex_ids[i] * 3, (vertex_ids[i] * 3) + 3);
		//basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
		basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(row_index, row_index + 3));
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		Mat submatrix_to_replace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affine_camera_matrix.copyTo(submatrix_to_replace);
	}

	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		y.at<float>(3 * i, 0) = landmarks[i][0];
		y.at<float>((3 * i) + 1, 0) = landmarks[i][1];
		//y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		//cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
		cv::Vec4f model_mean(face_instance.at<float>(vertex_ids[i]*3), face_instance.at<float>(vertex_ids[i]*3 + 1), face_instance.at<float>(vertex_ids[i]*3 + 2), 1.0f);
		v_bar.at<float>(4 * i, 0) = model_mean[0];
		v_bar.at<float>((4 * i) + 1, 0) = model_mean[1];
		v_bar.at<float>((4 * i) + 2, 0) = model_mean[2];
		//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
		// note: now that a Vec4f is returned, we could use copyTo?
	}

	// Bring into standard regularised quadratic form:
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
	
	Mat AtAReg = A.t() * A + lambda * Mat::eye(num_coeffs_to_fit, num_coeffs_to_fit, CV_32FC1);
	// Solve using OpenCV:
	Mat c_s;
	bool non_singular = cv::solve(AtAReg, -A.t() * b, c_s, cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
	// Because we're using SVD, non_singular will always be true. If we were to use e.g. Cholesky, we could return an expected<T>.

	return std::vector<float>(c_s);
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
 * @return The estimated blendshape-coefficients.
 */
inline std::vector<float> fit_blendshapes_to_landmarks_nnls(std::vector<eos::morphablemodel::Blendshape> blendshapes, cv::Mat face_instance, cv::Mat affine_camera_matrix, std::vector<cv::Vec2f> landmarks, std::vector<int> vertex_ids)
{
	using cv::Mat;
	assert(landmarks.size() == vertex_ids.size());

	int num_coeffs_to_fit = blendshapes.size();
	int num_landmarks = static_cast<int>(landmarks.size());

	// Copy all blendshapes into a "basis" matrix with each blendshape being a column:
	cv::Mat blendshapes_as_basis(blendshapes[0].deformation.rows, blendshapes.size(), CV_32FC1); // assert blendshapes.size() > 0 and all of them have same number of rows, and 1 col
	for (int i = 0; i < blendshapes.size(); ++i)
	{
		blendshapes[i].deformation.copyTo(blendshapes_as_basis.col(i));
	}

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		Mat basis_rows = blendshapes_as_basis.rowRange(vertex_ids[i] * 3, (vertex_ids[i] * 3) + 3);
		basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(row_index, row_index + 3)); // Todo: I think we can remove colRange here, as we always want to use all given blendshapes
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		Mat submatrix_to_replace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affine_camera_matrix.copyTo(submatrix_to_replace);
	}

	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		y.at<float>(3 * i, 0) = landmarks[i][0];
		y.at<float>((3 * i) + 1, 0) = landmarks[i][1];
		//y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		cv::Vec4f model_mean(face_instance.at<float>(vertex_ids[i]*3), face_instance.at<float>(vertex_ids[i]*3 + 1), face_instance.at<float>(vertex_ids[i]*3 + 2), 1.0f);
		v_bar.at<float>(4 * i, 0) = model_mean[0];
		v_bar.at<float>((4 * i) + 1, 0) = model_mean[1];
		v_bar.at<float>((4 * i) + 2, 0) = model_mean[2];
		//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
		// note: now that a Vec4f is returned, we could use copyTo?
	}

	// Bring into standard regularised quadratic form:
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.

	// Solve using NNLS:
	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> A_Eigen(A.ptr<float>(), A.rows, A.cols);
	Eigen::Map<RowMajorMatrixXf> b_Eigen(b.ptr<float>(), b.rows, b.cols);

	Eigen::VectorXf x;
	bool non_singular = Eigen::NNLS<Eigen::MatrixXf>::solve(A_Eigen, -b_Eigen, x);
	Mat c_s(x.rows(), x.cols(), CV_32FC1, x.data()); // create an OpenCV Mat header for the Eigen data

	
	return std::vector<float>(c_s);
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* BLENDSHAPEFITTING_HPP_ */
