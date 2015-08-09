/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/affine_camera_estimation.hpp
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

#ifndef AFFINECAMERAESTIMATION_HPP_
#define AFFINECAMERAESTIMATION_HPP_

#include "eos/render/utils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h" // for CV_REDUCE_AVG

#include <vector>
#include <cassert>

namespace eos {
	namespace fitting {

/**
 * The Gold Standard Algorithm for estimating an affine
 * camera matrix from world to image correspondences.
 * See Algorithm 7.2 in Multiple View Geometry, Hartley &
 * Zisserman, 2nd Edition, 2003.
 *
 * Requires >= 4 corresponding points.
 *
 * The estimated camera matrix works together with
 * render::render_affine(Mesh, cv::Mat, int, int, bool) to
 * for example render the model or extract texture from the
 * image.
 *
 * @param[in] image_points A list of 2D image points.
 * @param[in] model_points Corresponding points of a 3D model.
 * @return A 3x4 affine camera matrix (the third row is [0, 0, 0, 1]).
 */
cv::Mat estimate_affine_camera(std::vector<cv::Vec2f> image_points, std::vector<cv::Vec4f> model_points)
{
	using cv::Mat;
	assert(image_points.size() == model_points.size());
	assert(image_points.size() >= 4); // Number of correspondence points given needs to be equal to or larger than 4

	const int num_correspondences = static_cast<int>(image_points.size());

	Mat matImagePoints; // will be numCorrespondences x 2, CV_32FC1
	Mat matModelPoints; // will be numCorrespondences x 3, CV_32FC1

	for (int i = 0; i < image_points.size(); ++i) {
		Mat imgPoint(1, 2, CV_32FC1);
		imgPoint.at<float>(0, 0) = image_points[i][0];
		imgPoint.at<float>(0, 1) = image_points[i][1];
		matImagePoints.push_back(imgPoint);

		Mat mdlPoint(1, 3, CV_32FC1);
		mdlPoint.at<float>(0, 0) = model_points[i][0];
		mdlPoint.at<float>(0, 1) = model_points[i][1];
		mdlPoint.at<float>(0, 2) = model_points[i][2];
		matModelPoints.push_back(mdlPoint);
	}

	// translate the centroid of the image points to the origin:
	Mat imagePointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matImagePoints, imagePointsMean, 0, CV_REDUCE_AVG);
	imagePointsMean = cv::repeat(imagePointsMean, matImagePoints.rows, 1); // get T_13 and T_23 from imagePointsMean
	matImagePoints = matImagePoints - imagePointsMean;
	// scale the image points such that the RMS distance from the origin is sqrt(2):
	// 1) calculate the average norm (root mean squared distance) of all vectors
	float average_norm = 0.0f;
	for (int row = 0; row < matImagePoints.rows; ++row) {
		average_norm += static_cast<float>(cv::norm(matImagePoints.row(row), cv::NORM_L2));
	}
	average_norm /= matImagePoints.rows;
	// 2) multiply every vectors coordinate by sqrt(2)/average_norm
	float scaleFactor = static_cast<float>(std::sqrt(2)) / average_norm;
	matImagePoints *= scaleFactor; // add unit homogeneous component here
	// The points in matImagePoints now have a RMS distance from the origin of sqrt(2).
	// The normalisation matrix so that the 2D points are mean-free and their norm is as described above.
	Mat T = Mat::zeros(3, 3, CV_32FC1);
	T.at<float>(0, 0) = scaleFactor; // s_x
	T.at<float>(1, 1) = scaleFactor; // s_y
	T.at<float>(0, 2) = -imagePointsMean.at<float>(0, 0) * scaleFactor; // t_x
	T.at<float>(1, 2) = -imagePointsMean.at<float>(0, 1) * scaleFactor; // t_y
	T.at<float>(2, 2) = 1;

	// center the model points to the origin:
	Mat tmpOrigMdlPoints = matModelPoints.clone(); // Temp for testing: Save the original coordinates.
	// translate the centroid of the model points to the origin:
	Mat modelPointsMean; // use non-homogeneous coords for the next few steps? (less submatrices etc overhead)
	cv::reduce(matModelPoints, modelPointsMean, 0, CV_REDUCE_AVG);
	modelPointsMean = cv::repeat(modelPointsMean, matModelPoints.rows, 1);
	matModelPoints = matModelPoints - modelPointsMean;
	// scale the model points such that the RMS distance from the origin is sqrt(3):
	// 1) calculate the average norm (root mean squared distance) of all vectors
	average_norm = 0.0f;
	for (int row = 0; row < matModelPoints.rows; ++row) {
		average_norm += static_cast<float>(cv::norm(matModelPoints.row(row), cv::NORM_L2));
	}
	average_norm /= matModelPoints.rows;
	// 2) multiply every vectors coordinate by sqrt(3)/avgnorm
	scaleFactor = static_cast<float>(std::sqrt(3)) / average_norm;
	matModelPoints *= scaleFactor; // add unit homogeneous component here
	// The points in matModelPoints now have a RMS distance from the origin of sqrt(3).
	// The normalisation matrix so that the 3D points are mean-free and their norm is as described above.
	Mat U = Mat::zeros(4, 4, CV_32FC1);
	U.at<float>(0, 0) = scaleFactor; // s_x
	U.at<float>(1, 1) = scaleFactor; // s_y
	U.at<float>(2, 2) = scaleFactor; // s_z
	U.at<float>(0, 3) = -modelPointsMean.at<float>(0, 0) * scaleFactor; // t_x
	U.at<float>(1, 3) = -modelPointsMean.at<float>(0, 1) * scaleFactor; // t_y
	U.at<float>(2, 3) = -modelPointsMean.at<float>(0, 2) * scaleFactor; // t_z
	U.at<float>(3, 3) = 1;

	// Estimate the normalised camera matrix (C tilde).
	// We are solving the system $A_8 * p_8 = b$
	// The solution is obtained by the pseudo-inverse of A_8:
	// $p_8 = A_8^+ * b$
	Mat A_8 = Mat::zeros(num_correspondences * 2, 8, CV_32FC1);
	//Mat p_8(); // p_8 is 8 x 1. We are solving for it.
	Mat b(num_correspondences * 2, 1, CV_32FC1);
	for (int i = 0; i < num_correspondences; ++i) {
		A_8.at<float>(2 * i, 0) = matModelPoints.at<float>(i, 0); // could maybe made faster by assigning the whole row/col-range if possible?
		A_8.at<float>(2 * i, 1) = matModelPoints.at<float>(i, 1);
		A_8.at<float>(2 * i, 2) = matModelPoints.at<float>(i, 2);
		A_8.at<float>(2 * i, 3) = 1;
		A_8.at<float>((2 * i) + 1, 4) = matModelPoints.at<float>(i, 0);
		A_8.at<float>((2 * i) + 1, 5) = matModelPoints.at<float>(i, 1);
		A_8.at<float>((2 * i) + 1, 6) = matModelPoints.at<float>(i, 2);
		A_8.at<float>((2 * i) + 1, 7) = 1;
		b.at<float>(2 * i, 0) = matImagePoints.at<float>(i, 0);
		b.at<float>((2 * i) + 1, 0) = matImagePoints.at<float>(i, 1);
	}
	Mat p_8 = A_8.inv(cv::DECOMP_SVD) * b;
	Mat C_tilde = Mat::zeros(3, 4, CV_32FC1);
	C_tilde.row(0) = p_8.rowRange(0, 4).t(); // The first row of C_tilde consists of the first 4 entries of the col-vector p_8
	C_tilde.row(1) = p_8.rowRange(4, 8).t(); // Second row = last 4 entries
	C_tilde.at<float>(2, 3) = 1; // the last row is [0 0 0 1]

	Mat P_Affine = T.inv() * C_tilde * U;
	return P_Affine;
};

/**
 * Projects a point from world coordinates to screen coordinates.
 * First, an estimated affine camera matrix is used to transform
 * the point to clip space. Second, the point is transformed to
 * screen coordinates using the window transform. The window transform
 * also flips the y-axis (the image origin is top-left, while in
 * clip space top is +1 and bottom is -1).
 *
 * Note: Assumes the affine camera matrix only projects from world
 * to clip space, because a subsequent window transform is applied.
 * #Todo: This is outdated, now that we estimate the matrix from world
 * to screen space directly.
 *
 * @param[in] vertex A vertex in 3D space. vertex[3] = 1.0f.
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix.
 * @param[in] screen_width Width of the screen or window used for projection.
 * @param[in] screen_height Height of the screen or window used for projection.
 * @return A vector with x and y coordinates transformed to screen coordinates.
 */
inline cv::Vec2f project_affine(cv::Vec4f vertex, cv::Mat affine_camera_matrix, int screen_width, int screen_height)
{
	// Transform to clip space:
	cv::Mat clip_coords = affine_camera_matrix * cv::Mat(vertex);
	// Take the x and y coordinates in clip space and apply the window transform:
	cv::Vec2f screen_coords = render::clip_to_screen_space(cv::Vec2f(clip_coords.rowRange(0, 2)), screen_width, screen_height);
	return screen_coords;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* AFFINECAMERAESTIMATION_HPP_ */
