/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/AffineCameraEstimation.hpp
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

#include <vector>

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
 * @param[in] imagePoints A list of 2D image points.
 * @param[in] modelPoints Corresponding points of a 3D model.
 * @return A 3x4 affine camera matrix (the third row is [0, 0, 0, 1]).
 */
cv::Mat estimateAffineCamera(std::vector<cv::Vec2f> imagePoints, std::vector<cv::Vec4f> modelPoints);

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
