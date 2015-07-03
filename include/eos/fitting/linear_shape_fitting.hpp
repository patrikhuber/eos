/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
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

#include "eos/morphablemodel/MorphableModel.hpp"

#include "Eigen/LU"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <vector>
#include <cassert>

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
 * @param[in] morphable_model The Morphable Model whose shape (coefficients) are estimated.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix from world to clip-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image, given in clip-coordinates.
 * @param[in] vertex_ids The vertex ids in the model that correspond to the 2D points.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean).
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Not tested thoroughly.
 * @param[in] detector_standard_deviation The standard deviation of the 2D landmarks given (e.g. of the detector used).
 * @param[in] model_standard_deviation The standard deviation of the 3D vertex points in the 3D model.
 * @return The estimated shape-coefficients (alphas).
 */
inline std::vector<float> fit_shape_to_landmarks_linear(morphablemodel::MorphableModel morphable_model, cv::Mat affine_camera_matrix, std::vector<cv::Vec2f> landmarks, std::vector<int> vertex_ids, float lambda=20.0f, boost::optional<int> num_coefficients_to_fit=boost::optional<int>(), boost::optional<float> detector_standard_deviation=boost::optional<float>(), boost::optional<float> model_standard_deviation=boost::optional<float>())
{
	using cv::Mat;
	assert(landmarks.size() == vertex_ids.size());

	int num_coeffs_to_fit = num_coefficients_to_fit.get_value_or(morphable_model.get_shape_model().get_num_principal_components());
	int num_landmarks = static_cast<int>(landmarks.size());

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
	int row_index = 0;
	for (int i = 0; i < num_landmarks; ++i) {
		Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
																							//basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
		basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(row_index, row_index + 3));
		row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affineCam) is placed on the diagonal:
	Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
	for (int i = 0; i < num_landmarks; ++i) {
		Mat submatrix_to_replace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affine_camera_matrix.copyTo(submatrix_to_replace);
	}
	// The variances: Add the 2D and 3D standard deviations.
	// If the user doesn't provide them, we choose the following:
	// 2D (detector) variance: Assuming the detector has a standard deviation of 3 pixels, and the face size (IED) is around 80px. That's 3.75% of the IED. Assume that an image is on average 512x512px so 80/512 = 0.16 is the size the IED occupies inside an image.
	//                         Now we're in clip-coords ([-1, 1]) and take 0.16 of the range [-1, 1], 0.16/2 = 0.08, and then the standard deviation of the detector is 3.75% of 0.08, i.e. 0.0375*0.08 = 0.003.
	// 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
	float sigma_2D_3D = detector_standard_deviation.get_value_or(0.003f) + model_standard_deviation.get_value_or(0.0f);
	// Note: Isn't it a bit strange to add these as they have different units/normalisations? Check the paper.
	Mat Sigma = Mat::zeros(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
	for (int i = 0; i < 3 * num_landmarks; ++i) {
		Sigma.at<float>(i, i) = 1.0f / sigma_2D_3D; // the higher the sigma_2D_3D, the smaller the diagonal entries of Sigma will be
	}
	Mat Omega = Sigma.t() * Sigma; // just squares the diagonal
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
		cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
		v_bar.at<float>(4 * i, 0) = model_mean[0];
		v_bar.at<float>((4 * i) + 1, 0) = model_mean[1];
		v_bar.at<float>((4 * i) + 2, 0) = model_mean[2];
		//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
		// note: now that a Vec4f is returned, we could use copyTo?
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
	//Mat c_s; // The x, we solve for this! (the variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	//int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	int num_shape_pc = num_coeffs_to_fit;
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(num_shape_pc, num_shape_pc, CV_32FC1);
	// Invert using OpenCV:
	Mat AtOmegaARegInv = AtOmegaAReg.inv(cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
														  // Invert (and perform some sanity checks) using Eigen:
	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> AtOmegaAReg_Eigen(AtOmegaAReg.ptr<float>(), AtOmegaAReg.rows, AtOmegaAReg.cols);
	Eigen::FullPivLU<RowMajorMatrixXf> luOfAtOmegaAReg(AtOmegaAReg_Eigen); // Calculate the full-pivoting LU decomposition of the regularized AtA. Note: We could also try FullPivHouseholderQR if our system is non-minimal (i.e. there are more constraints than unknowns).
	auto rankOfAtOmegaAReg = luOfAtOmegaAReg.rank();
	bool isAtOmegaARegInvertible = luOfAtOmegaAReg.isInvertible();
	float threshold = /*2 * */ std::abs(luOfAtOmegaAReg.maxPivot()) * luOfAtOmegaAReg.threshold();
	RowMajorMatrixXf AtARegInv_EigenFullLU = luOfAtOmegaAReg.inverse(); // Note: We should use ::solve() instead
	Mat AtOmegaARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data

	Mat AtOmegatb = A.t() * Omega.t() * b;
	Mat c_s = -AtOmegaARegInv * AtOmegatb; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.

	return std::vector<float>(c_s);
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* LINEARSHAPEFITTING_HPP_ */
