/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/render_detail.hpp
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

#ifndef RENDER_DETAIL_HPP_
#define RENDER_DETAIL_HPP_

#include "opencv2/core/core.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
	namespace render {
		namespace detail {

/**
 * Takes a 3x4 affine camera matrix estimated with fitting::estimate_affine_camera
 * and computes the cross product of the first two rows to create a third axis that
 * is orthogonal to the first two.
 * This allows us to produce z values and figure out correct depth ordering in the
 * rendering and for texture extraction.
 *
 * @param[in] affine_camera_matrix A 3x4 affine camera matrix.
 * @return The matrix with a third row inserted.
 */
cv::Mat calculate_affine_z_direction(cv::Mat affine_camera_matrix)
{
	using cv::Mat;
	// Take the cross product of row 0 with row 1 to get the direction perpendicular to the viewing plane (= the viewing direction).
	// Todo: We should check if we look/project into the right direction - the sign could be wrong?
	Mat affine_cam_z_rotation = affine_camera_matrix.row(0).colRange(0, 3).cross(affine_camera_matrix.row(1).colRange(0, 3));
	affine_cam_z_rotation /= cv::norm(affine_cam_z_rotation, cv::NORM_L2);

	// The 4x4 affine camera matrix
	Mat affine_cam_4x4 = Mat::zeros(4, 4, CV_32FC1);

	// Replace the third row with the camera-direction (z)
	Mat third_row_rotation_part = affine_cam_4x4.row(2).colRange(0, 3);
	affine_cam_z_rotation.copyTo(third_row_rotation_part); // Set first 3 components. 4th component stays 0.

	// Copy the first 2 rows from the input matrix
	Mat first_two_rows_of_4x4 = affine_cam_4x4.rowRange(0, 2);
	affine_camera_matrix.rowRange(0, 2).copyTo(first_two_rows_of_4x4);

	// The 4th row is (0, 0, 0, 1):
	affine_cam_4x4.at<float>(3, 3) = 1.0f;

	return affine_cam_4x4;
};

/**
 * Just a representation for a vertex during rendering.
 *
 * Might consider getting rid of it.
 */
class Vertex
{
public:
	Vertex() {};
	Vertex(const cv::Vec4f& position, const cv::Vec3f& color, const cv::Vec2f& texCoord) : position(position), color(color), texcrd(texCoord) {};

	cv::Vec4f position;
	cv::Vec3f color;
	cv::Vec2f texcrd;
};

/**
 * A representation for a triangle that is to be rasterised.
 * Stores the enclosing bounding box of the triangle that is
 * calculated during rendering and used during rasterisation.
 */
struct TriangleToRasterize
{
	Vertex v0, v1, v2;
	int min_x;
	int max_x;
	int min_y;
	int max_y;
};

/**
 * Calculates the enclosing bounding box of 3 vertices (a triangle). If the
 * triangle is partly outside the screen, it will be clipped appropriately.
 *
 * Todo: If it is fully outside the screen, check what happens, but it works.
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @return A bounding box rectangle.
 */
cv::Rect calculate_clipped_bounding_box(cv::Vec4f v0, cv::Vec4f v1, cv::Vec4f v2, int viewport_width, int viewport_height)
{
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;
	int minX = max(min(floor(v0[0]), min(floor(v1[0]), floor(v2[0]))), 0.0f);
	int maxX = min(max(ceil(v0[0]), max(ceil(v1[0]), ceil(v2[0]))), static_cast<float>(viewport_width - 1));
	int minY = max(min(floor(v0[1]), min(floor(v1[1]), floor(v2[1]))), 0.0f);
	int maxY = min(max(ceil(v0[1]), max(ceil(v1[1]), ceil(v2[1]))), static_cast<float>(viewport_height - 1));
	return cv::Rect(minX, minY, maxX - minX, maxY - minY);
};

/**
 * Computes whether the triangle formed out of the given three vertices is
 * counter-clockwise in screen space. Assumes the origin of the screen is on
 * the top-left, and the y-axis goes down (as in OpenCV images).
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return Whether the vertices are CCW in screen space.
 */
bool are_vertices_ccw_in_screen_space(const cv::Vec4f& v0, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	float dx01 = v1[0] - v0[0];
	float dy01 = v1[1] - v0[1];
	float dx02 = v2[0] - v0[0];
	float dy02 = v2[1] - v0[1];

	return (dx01*dy02 - dy01*dx02 < 0.0f); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
};

double implicit_line(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
};

/**
 * Rasters a triangle into the given colour and depth buffer.
 *
 * In essence, loop through the pixels inside the triangle's bounding
 * box, calculate the barycentric coordinates, and if inside the triangle
 * and the z-test is passed, then draw the point using the barycentric
 * coordinates for colour interpolation.

 * Does not do perspective-correct weighting, and therefore only works
 * with the affine rendering pipeline.
 *
 * No texturing at the moment.
 *
 * @param[in] triangle A triangle.
 * @param[in] colourbuffer The colour buffer to draw into.
 * @param[in] depthbuffer The depth buffer to draw into and use for the depth test.
 */
void raster_triangle(TriangleToRasterize triangle, cv::Mat colourbuffer, cv::Mat depthbuffer)
{
	for (int yi = triangle.min_y; yi <= triangle.max_y; yi++)
	{
		for (int xi = triangle.min_x; xi <= triangle.max_x; xi++)
		{
			// we want centers of pixels to be used in computations. Do we?
			const float x = static_cast<float>(xi) + 0.5f;
			const float y = static_cast<float>(yi) + 0.5f;

			// these will be used for barycentric weights computation
			const double one_over_v0ToLine12 = 1.0 / implicit_line(triangle.v0.position[0], triangle.v0.position[1], triangle.v1.position, triangle.v2.position);
			const double one_over_v1ToLine20 = 1.0 / implicit_line(triangle.v1.position[0], triangle.v1.position[1], triangle.v2.position, triangle.v0.position);
			const double one_over_v2ToLine01 = 1.0 / implicit_line(triangle.v2.position[0], triangle.v2.position[1], triangle.v0.position, triangle.v1.position);
			// affine barycentric weights
			const double alpha = implicit_line(x, y, triangle.v1.position, triangle.v2.position) * one_over_v0ToLine12;
			const double beta = implicit_line(x, y, triangle.v2.position, triangle.v0.position) * one_over_v1ToLine20;
			const double gamma = implicit_line(x, y, triangle.v0.position, triangle.v1.position) * one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				const int pixel_index_row = yi;
				const int pixel_index_col = xi;

				const double z_affine = alpha*static_cast<double>(triangle.v0.position[2]) + beta*static_cast<double>(triangle.v1.position[2]) + gamma*static_cast<double>(triangle.v2.position[2]);
				if (z_affine < depthbuffer.at<double>(pixel_index_row, pixel_index_col))
				{
					// attributes interpolation
					// pixel_color is in RGB, v.color are RGB
					cv::Vec3f pixel_color = alpha*triangle.v0.color + beta*triangle.v1.color + gamma*triangle.v2.color;

					// clamp bytes to 255
					const unsigned char red = static_cast<unsigned char>(255.0f * std::min(pixel_color[0], 1.0f)); // Todo: Proper casting (rounding?)
					const unsigned char green = static_cast<unsigned char>(255.0f * std::min(pixel_color[1], 1.0f));
					const unsigned char blue = static_cast<unsigned char>(255.0f * std::min(pixel_color[2], 1.0f));

					// update buffers
					colourbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[0] = blue;
					colourbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[1] = green;
					colourbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[2] = red;
					colourbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[3] = 255; // alpha channel
					depthbuffer.at<double>(pixel_index_row, pixel_index_col) = z_affine;
				}
			}
		}
	}
};

		} /* namespace detail */
	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_DETAIL_HPP_ */
