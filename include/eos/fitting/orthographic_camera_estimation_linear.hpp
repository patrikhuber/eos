/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/orthographic_camera_estimation_linear.hpp
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

#ifndef ORTHOGRAPHICCAMERAESTIMATIONLINEAR_HPP_
#define ORTHOGRAPHICCAMERAESTIMATIONLINEAR_HPP_

#include "glm/mat3x3.hpp"

#include "opencv2/core/core.hpp" // Remove eventually
#include "Eigen/Core"
#include "Eigen/SVD"

#include "boost/optional.hpp"

#include <vector>
#include <cassert>

namespace eos {
	namespace fitting {

/**
 * Parameters of an estimated scaled orthographic projection.
 */
struct ScaledOrthoProjectionParameters {
	glm::mat3x3 R;
	double tx, ty;
	double s;
};

/**
 * Estimates the parameters of a scaled orthographic projection.
 *
 * Given a set of 2D-3D correspondences, this algorithm estimates rotation,
 * translation (in x and y) and a scaling parameters of the scaled orthographic
 * projection model using a closed-form solution. It does so by first computing
 * an affine camera matrix using algorithm [1], and then finds the closest
 * orthonormal matrix to the estimated affine transform using SVD.
 * This algorithm follows the original implementation [2] of William Smith,
 * University of York.
 *
 * Requires >= 4 corresponding points.
 *
 * [1]: Gold Standard Algorithm for estimating an affine camera matrix from
 * world to image correspondences, Algorithm 7.2 in Multiple View Geometry,
 * Hartley & Zisserman, 2nd Edition, 2003.
 * [2]: https://github.com/waps101/3DMM_edges/blob/master/utils/POS.m
 *
 * @param[in] image_points A list of 2D image points.
 * @param[in] model_points Corresponding points of a 3D model.
 * @param[in] is_viewport_upsidedown Flag to set whether the viewport of the image points is upside-down (e.g. as in OpenCV).
 * @param[in] viewport_height Height of the viewport of the image points (needs to be given if is_viewport_upsidedown == true).
 * @return Rotation, translation and scaling of the estimated scaled orthographic projection.
 */
inline ScaledOrthoProjectionParameters estimate_orthographic_projection_linear(std::vector<Eigen::Vector2f> image_points, std::vector<Eigen::Vector4f> model_points, bool is_viewport_upsidedown, boost::optional<int> viewport_height = boost::none)
{
	using cv::Mat;
	using Eigen::Matrix;
	assert(image_points.size() == model_points.size());
	assert(image_points.size() >= 4); // Number of correspondence points given needs to be equal to or larger than 4

	const int num_correspondences = static_cast<int>(image_points.size());

	if (is_viewport_upsidedown)
	{
		if (viewport_height == boost::none)
		{
			throw std::runtime_error("Error: If is_viewport_upsidedown is set to true, viewport_height needs to be given.");
		}
		for (auto&& ip : image_points)
		{
			ip[1] = viewport_height.get() - ip[1];
		}
	}

	Mat A_ = Mat::zeros(2 * num_correspondences, 8, CV_32FC1);
	Matrix<float, Eigen::Dynamic, 8> A = Matrix<float, Eigen::Dynamic, 8>::Zero(2 * num_correspondences, 8);
	int row_index = 0;
	for (int i = 0; i < model_points.size(); ++i)
	{
		cv::Vec4f tmp(model_points[i][0], model_points[i][1], model_points[i][2], model_points[i][3]); // Temp, remove! Switch to Eigen.
		Mat p = Mat(tmp).t();
		p.copyTo(A_.row(row_index).colRange(0, 4)); // even row - copy to left side (first row is row 0)
		A.block<1, 4>(row_index, 0) = model_points[i]; // no transpose?
		row_index++;
		p.copyTo(A_.row(row_index).colRange(4, 8)); // odd row - copy to right side
		A.block<1, 4>(row_index, 4) = model_points[i];
		row_index++;
	} // 4th coord (homogeneous) is already 1

	Mat b_(2 * num_correspondences, 1, CV_32FC1);
	Matrix<float, Eigen::Dynamic, 1> b(2 * num_correspondences, 1);
	row_index = 0;
	for (int i = 0; i < image_points.size(); ++i)
	{
		b.segment<2>(row_index) = image_points[i];
		b_.at<float>(row_index) = image_points[i][0];
		row_index++;
		b_.at<float>(row_index) = image_points[i][1];
		row_index++;
	}

	Mat k_; // resulting affine matrix (8x1)
	bool solved = cv::solve(A_, b_, k_, cv::DECOMP_SVD);
	// The original implementation used SVD here to solve the linear system, but
	// QR seems to do the job fine too.
	const Matrix<float, 8, 1> k = A.colPivHouseholderQr().solve(b);

	const Mat R1_ = k_.rowRange(0, 3);
	const Eigen::Vector3f R1 = k.segment<3>(0);
	const Mat R2_ = k_.rowRange(4, 7);
	const Eigen::Vector3f R2 = k.segment<3>(4);
	const float sTx_ = k_.at<float>(3);
	const float sTx = k(3);
	const float sTy_ = k_.at<float>(7);
	const float sTy = k(7);
	const auto s_ = (cv::norm(R1_) + cv::norm(R2_)) / 2.0;
	const auto s = (R1.norm() + R2.norm()) / 2.0;
	Mat r1_ = R1_ / cv::norm(R1_);
	Mat r2_ = R2_ / cv::norm(R2_);
	Mat r3_ = r1_.cross(r2_);
	Mat R_;
	Eigen::Matrix3f R;
	Eigen::Vector3f r1 = R1.normalized(); // Not sure why R1.normalize() (in-place) produces a compiler error.
	Eigen::Vector3f r2 = R2.normalized();
	R.block<1, 3>(0, 0) = r1;
	R.block<1, 3>(1, 0) = r2;
	R.block<1, 3>(2, 0) = r1.cross(r2);
	r1_ = r1_.t();
	r2_ = r2_.t();
	r3_ = r3_.t();
	R_.push_back(r1_);
	R_.push_back(r2_);
	R_.push_back(r3_);
	// Set R to the closest orthonormal matrix to the estimated affine transform:
	Mat S_, U_, Vt_;
	cv::SVDecomp(R_, S_, U_, Vt_);
	Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::NoQRPreconditioner> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU();
	const Eigen::Matrix3f V = svd.matrixV();
	Mat R_ortho_ = U_ * Vt_;
	Eigen::Matrix3f R_ortho = U * V.transpose();
	// The determinant of R must be 1 for it to be a valid rotation matrix
	if (cv::determinant(R_ortho_) < 0)
	{
		U_.row(2) = -U_.row(2); // not sure this works...
		R_ortho_ = U_ * Vt_;
	}
	if (R_ortho.determinant() < 0)
	{
		U.block<1, 3>(2, 0) = -U.block<1, 3>(2, 0);
		R_ortho = U * V.transpose();
	}

	// Remove the scale from the translations:
	const auto t1_ = sTx_ / s_;
	const auto t2_ = sTy_ / s_;
	const auto t1 = sTx / s;
	const auto t2 = sTy / s;

	// Convert to a glm::mat4x4:
	glm::mat3x3 R_glm_; // identity
	glm::mat3x3 R_glm;
	for (int r = 0; r < 3; ++r) {
		for (int c = 0; c < 3; ++c) {
			R_glm_[c][r] = R_ortho_.at<float>(r, c);
			R_glm[c][r] = R_ortho(r, c);
		}
	}
	return ScaledOrthoProjectionParameters{ R_glm, t1_, t2_, s_ };
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* ORTHOGRAPHICCAMERAESTIMATIONLINEAR_HPP_ */
