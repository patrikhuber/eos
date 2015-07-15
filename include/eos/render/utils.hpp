/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/utils.hpp
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

#ifndef RENDER_UTILS_HPP_
#define RENDER_UTILS_HPP_

#include "opencv2/core/core.hpp"

namespace eos {
	namespace render {

/**
 * Transforms a point from clip space ([-1, 1] x [-1, 1]) to
 * image (screen) coordinates, i.e. the window transform.
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 * No z-division is performed.
 * Note: It should rather be called from NDC to screen space?
 *
 * Exactly conforming to the OpenGL viewport transform, except that
 * we flip y at the end.
 * Qt: Origin top-left. OpenGL: bottom-left. OCV: top-left.
 *
 * @param[in] clip_coordinates A point in clip coordinates.
 * @param[in] screen_width Width of the screen or window.
 * @param[in] screen_height Height of the screen or window.
 * @return A vector with x and y coordinates transformed to screen space.
 */
inline cv::Vec2f clip_to_screen_space(const cv::Vec2f& clip_coordinates, int screen_width, int screen_height)
{
	// Window transform:
	float x_ss = (clip_coordinates[0] + 1.0f) * (screen_width / 2.0f);
	float y_ss = screen_height - (clip_coordinates[1] + 1.0f) * (screen_height / 2.0f); // also flip y; Qt: Origin top-left. OpenGL: bottom-left.
	return cv::Vec2f(x_ss, y_ss);
	/* Note: What we do here is equivalent to
	   x_w = (x *  vW/2) + vW/2;
	   However, Shirley says we should do:
	   x_w = (x *  vW/2) + (vW-1)/2;
	   (analogous for y)
	   Todo: Check the consequences.
	*/
};

/**
 * Transforms a point from image (screen) coordinates to
 * clip space ([-1, 1] x [-1, 1]).
 * Note that the y-coordinate is flipped because the image origin
 * is top-left while in clip space top is +1 and bottom is -1.
 *
 * @param[in] screenCoordinates A point in screen coordinates.
 * @param[in] screenWidth Width of the screen or window.
 * @param[in] screenHeight Height of the screen or window.
 * @return A vector with x and y coordinates transformed to clip space.
 */
inline cv::Vec2f screen_to_clip_space(const cv::Vec2f& screen_coordinates, int screen_width, int screen_height)
{
	float x_cs = screen_coordinates[0] / (screen_width / 2.0f) - 1.0f;
	float y_cs = screen_coordinates[1] / (screen_height / 2.0f) - 1.0f;
	y_cs *= -1.0f;
	return cv::Vec2f(x_cs, y_cs);
};

inline cv::Rect calculate_bounding_box(cv::Vec4f v0, cv::Vec4f v1, cv::Vec4f v2, int viewportWidth, int viewportHeight)
{
	/* Old, producing artifacts:
	t.minX = max(min(t.v0.position[0], min(t.v1.position[0], t.v2.position[0])), 0.0f);
	t.maxX = min(max(t.v0.position[0], max(t.v1.position[0], t.v2.position[0])), (float)(viewportWidth - 1));
	t.minY = max(min(t.v0.position[1], min(t.v1.position[1], t.v2.position[1])), 0.0f);
	t.maxY = min(max(t.v0.position[1], max(t.v1.position[1], t.v2.position[1])), (float)(viewportHeight - 1));*/

	int minX = std::max(std::min(floor(v0[0]), std::min(floor(v1[0]), floor(v2[0]))), 0.0f);
	int maxX = std::min(std::max(ceil(v0[0]), std::max(ceil(v1[0]), ceil(v2[0]))), static_cast<float>(viewportWidth - 1));
	int minY = std::max(std::min(floor(v0[1]), std::min(floor(v1[1]), floor(v2[1]))), 0.0f);
	int maxY = std::min(std::max(ceil(v0[1]), std::max(ceil(v1[1]), ceil(v2[1]))), static_cast<float>(viewportHeight - 1));
	return cv::Rect(minX, minY, maxX - minX, maxY - minY);
};

// Returns true if inside the tri or on the border
inline bool is_point_in_triangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	/* See http://www.blackpawn.com/texts/pointinpoly/ */
	// Compute vectors        
	cv::Point2f v0 = triV2 - triV0;
	cv::Point2f v1 = triV1 - triV0;
	cv::Point2f v2 = point - triV0;

	// Compute dot products
	float dot00 = v0.dot(v0);
	float dot01 = v0.dot(v1);
	float dot02 = v0.dot(v2);
	float dot11 = v1.dot(v1);
	float dot12 = v1.dot(v2);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v < 1);
};

inline bool are_vertices_cc_in_screen_space(const cv::Vec4f& v0, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	float dx01 = v1[0] - v0[0];
	float dy01 = v1[1] - v0[1];
	float dx02 = v2[0] - v0[0];
	float dy02 = v2[1] - v0[1];

	return (dx01*dy02 - dy01*dx02 < 0.0f); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
};

inline double implicit_line(float x, float y, const cv::Vec4f& v1, const cv::Vec4f& v2)
{
	return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
};

// enum of transformation types used in mapping
enum class MappingTransformation {
	NN,
	BILINEAR,
	AREA
};

/**
* extracts the texture of the face from the image
*
*
* @param[in] mesh TODO
* @param[in] mvpMatrix Atm working with a 4x4 (full) affine. But anything would work, just take care with the w-division.
* @param[in] view_port_width TODO
* @param[in] view_port_height TODO
* @param[in] image Where to extract the texture from
* @param[in] depth_buffer TODO note: We could also pass an instance of a Renderer here. Depending on how "stateful" the renderer is, this might make more sense.
* @param[in] mapping_type Which Transformation type to use for mapping
* @return A Mat with the texture as an isomap
* // note: framebuffer should have size of the image (ok not necessarily. What about mobile?) (well it should, to get optimal quality (and everywhere the same quality)?)
*/
inline cv::Mat extract_texture(Mesh mesh, cv::Mat mvpMatrix, int view_port_width, int view_port_height, cv::Mat image, cv::Mat depth_buffer, MappingTransformation mapping_type) {

	using cv::Mat;
	using cv::Vec4f;
	using cv::Vec2f;

	// optional param  cv::Mat texture_map = Mat(512, 512, CV_8UC3) ?
	// cv::Mat texture_map(512, 512, inputImage.type());
	Mat texture_map = Mat::zeros(512, 512, CV_8UC3); // We don't want an alpha channel. We might want to handle grayscale input images though.
	Mat visibilityMask = Mat::zeros(512, 512, CV_8UC3);


	for (const auto& triangleIndices : mesh.tvi) {

		// Find out if the current triangle is visible:
		// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
		// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
		// the texture.
		// Possible improvement: - If only part of the triangle is visible, split it
		// - Share more code with the renderer?
		Vec4f v0_3d = mesh.vertices[triangleIndices[0]];
		Vec4f v1_3d = mesh.vertices[triangleIndices[1]];
		Vec4f v2_3d = mesh.vertices[triangleIndices[2]];

		Vec4f v0, v1, v2; // we don't copy the color and texcoords, we only do the visibility check here.
		// This could be optimized in 2 ways though:
		// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
		// - We transform them later (below) a second time. Only do it once.
		v0 = Mat(mvpMatrix * Mat(v0_3d));
		v1 = Mat(mvpMatrix * Mat(v1_3d));
		v2 = Mat(mvpMatrix * Mat(v2_3d));

		// Well, in in principle, we'd have to do the whole stuff as in render(), like
		// clipping against the frustums etc.
		// But as long as our model is fully on the screen, we're fine.

		// divide by w
		// if ortho, we can do the divide as well, it will just be a / 1.0f.
		v0 = v0 / v0[3];
		v1 = v1 / v1[3];
		v2 = v2 / v2[3];

		// Todo: This is all very similar to processProspectiveTri(...), except the other function does texturing stuff as well. Remove code duplication!
		Vec2f v0_clip = clip_to_screen_space(Vec2f(v0[0], v0[1]), view_port_width, view_port_height);
		Vec2f v1_clip = clip_to_screen_space(Vec2f(v1[0], v1[1]), view_port_width, view_port_height);
		Vec2f v2_clip = clip_to_screen_space(Vec2f(v2[0], v2[1]), view_port_width, view_port_height);
		v0[0] = v0_clip[0]; v0[1] = v0_clip[1];
		v1[0] = v1_clip[0]; v1[1] = v1_clip[1];
		v2[0] = v2_clip[0]; v2[1] = v2_clip[1];

		//if (doBackfaceCulling) {
		if (!are_vertices_cc_in_screen_space(v0, v1, v2))
			continue;
		//}

		cv::Rect bbox = calculate_bounding_box(v0, v1, v2, view_port_width, view_port_height);
		int minX = bbox.x;
		int maxX = bbox.x + bbox.width;
		int minY = bbox.y;
		int maxY = bbox.y + bbox.height;

		//if (t.maxX <= t.minX || t.maxY <= t.minY) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		//	continue;

		bool wholeTriangleIsVisible = true;
		for (int yi = minY; yi <= maxY; yi++)
		{
			for (int xi = minX; xi <= maxX; xi++)
			{
				// we want centers of pixels to be used in computations. TODO: Do we?
				float x = (float)xi + 0.5f;
				float y = (float)yi + 0.5f;
				// these will be used for barycentric weights computation
				double one_over_v0ToLine12 = 1.0 / implicit_line(v0[0], v0[1], v1, v2);
				double one_over_v1ToLine20 = 1.0 / implicit_line(v1[0], v1[1], v2, v0);
				double one_over_v2ToLine01 = 1.0 / implicit_line(v2[0], v2[1], v0, v1);
				// affine barycentric weights
				double alpha = implicit_line(x, y, v1, v2) * one_over_v0ToLine12;
				double beta = implicit_line(x, y, v2, v0) * one_over_v1ToLine20;
				double gamma = implicit_line(x, y, v0, v1) * one_over_v2ToLine01;
				// if pixel (x, y) is inside the triangle or on one of its edges
				if (alpha >= 0 && beta >= 0 && gamma >= 0)
				{
					double z_affine = alpha*(double)v0[2] + beta*(double)v1[2] + gamma*(double)v2[2];
					// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
					if (z_affine < depth_buffer.at<double>(yi, xi)/* && z_affine <= 1.0*/) {
						wholeTriangleIsVisible = false;
						break;
					}
					else {

					}
				}
			}
			if (!wholeTriangleIsVisible) {
				break;
			}
		}

		if (!wholeTriangleIsVisible) {
			continue;
		}

		cv::Point2f srcTri[3];
		cv::Point2f dstTri[3];
		cv::Vec4f vec(mesh.vertices[triangleIndices[0]][0], mesh.vertices[triangleIndices[0]][1], mesh.vertices[triangleIndices[0]][2], 1.0f);
		cv::Vec4f res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		Vec2f screenSpace = clip_to_screen_space(Vec2f(res[0], res[1]), view_port_width, view_port_height);
		srcTri[0] = screenSpace;

		vec = cv::Vec4f(mesh.vertices[triangleIndices[1]][0], mesh.vertices[triangleIndices[1]][1], mesh.vertices[triangleIndices[1]][2], 1.0f);
		res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		screenSpace = clip_to_screen_space(Vec2f(res[0], res[1]), view_port_width, view_port_height);
		srcTri[1] = screenSpace;

		vec = cv::Vec4f(mesh.vertices[triangleIndices[2]][0], mesh.vertices[triangleIndices[2]][1], mesh.vertices[triangleIndices[2]][2], 1.0f);
		res = Mat(mvpMatrix * Mat(vec));
		res /= res[3];
		screenSpace = clip_to_screen_space(Vec2f(res[0], res[1]), view_port_width, view_port_height);
		srcTri[2] = screenSpace;

		dstTri[0] = cv::Point2f(texture_map.cols*mesh.texcoords[triangleIndices[0]][0], texture_map.rows*mesh.texcoords[triangleIndices[0]][1] - 1.0f);
		dstTri[1] = cv::Point2f(texture_map.cols*mesh.texcoords[triangleIndices[1]][0], texture_map.rows*mesh.texcoords[triangleIndices[1]][1] - 1.0f);
		dstTri[2] = cv::Point2f(texture_map.cols*mesh.texcoords[triangleIndices[2]][0], texture_map.rows*mesh.texcoords[triangleIndices[2]][1] - 1.0f);


		// Get the inverse Affine Transform from original image: from dst to src
		cv::Mat warp_mat_org_inv = cv::getAffineTransform(dstTri, srcTri);
		warp_mat_org_inv.convertTo(warp_mat_org_inv, CV_32FC1);

		// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
		for (int x = std::min(dstTri[0].x, std::min(dstTri[1].x, dstTri[2].x)); x < std::max(dstTri[0].x, std::max(dstTri[1].x, dstTri[2].x)); ++x) {
			for (int y = std::min(dstTri[0].y, std::min(dstTri[1].y, dstTri[2].y)); y < std::max(dstTri[0].y, std::max(dstTri[1].y, dstTri[2].y)); ++y) {
				if (is_point_in_triangle(cv::Point2f(x, y), dstTri[0], dstTri[1], dstTri[2])) {


					if (mapping_type == MappingTransformation::AREA){

						//calculate positions of 4 corners of pixel in image (src)
						cv::Vec3f homogenous_dst_upper_left = cv::Vec3f(x - 0.5, y - 0.5, 1.f);
						cv::Vec3f homogenous_dst_upper_right = cv::Vec3f(x + 0.5, y - 0.5, 1.f);
						cv::Vec3f homogenous_dst_lower_left = cv::Vec3f(x - 0.5, y + 0.5, 1.f);
						cv::Vec3f homogenous_dst_lower_right = cv::Vec3f(x + 0.5, y + 0.5, 1.f);

						cv::Vec2f srcTexel_upper_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_left));
						cv::Vec2f srcTexel_upper_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_right));
						cv::Vec2f srcTexel_lower_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_left));
						cv::Vec2f srcTexel_lower_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_right));

						float min_a = std::min(std::min(srcTexel_upper_left[0], srcTexel_upper_right[0]), std::min(srcTexel_lower_left[0], srcTexel_lower_right[0]));
						float max_a = std::max(std::max(srcTexel_upper_left[0], srcTexel_upper_right[0]), std::max(srcTexel_lower_left[0], srcTexel_lower_right[0]));
						float min_b = std::min(std::min(srcTexel_upper_left[1], srcTexel_upper_right[1]), std::min(srcTexel_lower_left[1], srcTexel_lower_right[1]));
						float max_b = std::max(std::max(srcTexel_upper_left[1], srcTexel_upper_right[1]), std::max(srcTexel_lower_left[1], srcTexel_lower_right[1]));

						cv::Vec3i color = cv::Vec3i();
						int num_texels = 0;

						for (int a = ceil(min_a); a <= floor(max_a); ++a)
						{
							for (int b = ceil(min_b); b <= floor(max_b); ++b)
							{
								if (is_point_in_triangle(cv::Point2f(a, b), srcTexel_upper_left, srcTexel_lower_left, srcTexel_upper_right) || is_point_in_triangle(cv::Point2f(a, b), srcTexel_lower_left, srcTexel_upper_right, srcTexel_lower_right)) {
									if (a < image.cols && b < image.rows){ //if srcTexel in triangle and in image
										num_texels++;
										color += image.at<cv::Vec3b>(b, a);
									}
								}

							}
						}
						if (num_texels > 0)
							color = color / num_texels;
						else{ //if no corresponding texel found, nearest neighbor interpolation
							//calculate corrresponding position of dst_coord pixel center in image (src)
							cv::Vec3f homogenous_dst_coord = cv::Vec3f(x, y, 1.f);
							cv::Vec2f srcTexel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

							if ((cvRound(srcTexel[1]) < image.rows) && cvRound(srcTexel[0]) < image.cols) {
								color = image.at<cv::Vec3b>(cvRound(srcTexel[1]), cvRound(srcTexel[0]));
							}


						}
						texture_map.at<cv::Vec3b>(y, x) = color;
					}
					else if (mapping_type == MappingTransformation::BILINEAR){

						//calculate corrresponding position of dst_coord pixel center in image (src)
						cv::Vec3f homogenous_dst_coord = cv::Vec3f(x, y, 1.f);
						cv::Vec2f srcTexel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						//calculate distances to next 4 pixels
						float distance_upper_left = sqrt(powf(srcTexel[0] - floor(srcTexel[0]), 2) + powf(srcTexel[1] - floor(srcTexel[1]), 2));
						float distance_upper_right = sqrt(powf(srcTexel[0] - floor(srcTexel[0]), 2) + powf(srcTexel[1] - ceil(srcTexel[1]), 2));
						float distance_lower_left = sqrt(powf(srcTexel[0] - ceil(srcTexel[0]), 2) + powf(srcTexel[1] - floor(srcTexel[1]), 2));
						float distance_lower_right = sqrt(powf(srcTexel[0] - ceil(srcTexel[0]), 2) + powf(srcTexel[1] - ceil(srcTexel[1]), 2));

						//normalize distances
						float sum_distances = distance_lower_left + distance_lower_right + distance_upper_left + distance_upper_right;
						distance_lower_left /= sum_distances;
						distance_lower_right /= sum_distances;
						distance_upper_left /= sum_distances;
						distance_upper_right /= sum_distances;

						// set color depending on distance from next 4 pixels
						for (int color = 0; color < 3; color++){
							float color_upper_left = image.at<cv::Vec3b>(floor(srcTexel[1]), floor(srcTexel[0]))[color] * distance_upper_left;
							float color_upper_right = image.at<cv::Vec3b>(floor(srcTexel[1]), ceil(srcTexel[0]))[color] * distance_upper_right;
							float color_lower_left = image.at<cv::Vec3b>(ceil(srcTexel[1]), floor(srcTexel[0]))[color] * distance_lower_left;
							float color_lower_right = image.at<cv::Vec3b>(ceil(srcTexel[1]), ceil(srcTexel[0]))[color] * distance_lower_right;

							texture_map.at<cv::Vec3b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
						}

					}
					else if (mapping_type == MappingTransformation::NN){

						//calculate corrresponding position of dst_coord pixel center in image (src)
						cv::Vec3f homogenous_dst_coord = cv::Vec3f(x, y, 1.f);
						cv::Vec2f srcTexel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						if ((cvRound(srcTexel[1]) < image.rows) && (cvRound(srcTexel[0]) < image.cols))
							texture_map.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(cvRound(srcTexel[1]), cvRound(srcTexel[0]));

					}



				}
			}
		}


	}
	return texture_map;
};

	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_UTILS_HPP_ */
