/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/texture_extraction.hpp
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

#ifndef TEXTURE_EXTRACTION_HPP_
#define TEXTURE_EXTRACTION_HPP_

#include "eos/render/Mesh.hpp"
#include "eos/render/detail/render_detail.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace eos {
	namespace render {

/**
 * Computes whether the given point is inside (or on the border of) the triangle
 * formed out of the given three vertices.
 *
 * @param[in] point Point to check.
 * @param[in] triV0 First vertex.
 * @param[in] triV1 Second vertex.
 * @param[in] triV2 Third vertex.
 * @return Whether the point is inside the triangle.
 */
inline bool is_point_in_triangle(cv::Point2f point, cv::Point2f triV0, cv::Point2f triV1, cv::Point2f triV2) {
	// See http://www.blackpawn.com/texts/pointinpoly/
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

/**
 * The interpolation types that can be used to map the
 * texture from the original image to the isomap.
 */
enum class TextureInterpolation {
	NearestNeighbour,
	Bilinear,
	Area
};

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] mvp_matrix An estimated 3x4 affine camera matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] image The image to extract the texture from.
 * @param[in] depth_buffer A precalculated depthbuffer image.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @return The extracted texture as isomap (texture map).
 */
inline cv::Mat extract_texture(Mesh mesh, cv::Mat mvp_matrix, int viewport_width, int viewport_height, cv::Mat image, cv::Mat depth_buffer, TextureInterpolation mapping_type)
{
	using cv::Mat;
	using cv::Vec2f;
	using cv::Vec3f;
	using cv::Vec4f;
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;

	mvp_matrix = detail::calculate_affine_z_direction(mvp_matrix);

	// optional param  cv::Mat texture_map = Mat(512, 512, CV_8UC3) ?
	// cv::Mat texture_map(512, 512, inputImage.type());
	Mat texture_map = Mat::zeros(512, 512, CV_8UC3); // We don't want an alpha channel. We might want to handle grayscale input images though.

	for (const auto& triangle_indices : mesh.tvi) {

		// Find out if the current triangle is visible:
		// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
		// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
		// the texture.
		// Possible improvement: - If only part of the triangle is visible, split it
		// - Share more code with the renderer?
		Vec4f v0_3d = mesh.vertices[triangle_indices[0]];
		Vec4f v1_3d = mesh.vertices[triangle_indices[1]];
		Vec4f v2_3d = mesh.vertices[triangle_indices[2]];

		Vec4f v0, v1, v2; // we don't copy the color and texcoords, we only do the visibility check here.
		// This could be optimized in 2 ways though:
		// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
		// - We transform them later (below) a second time. Only do it once.
		v0 = Mat(mvp_matrix * Mat(v0_3d));
		v1 = Mat(mvp_matrix * Mat(v1_3d));
		v2 = Mat(mvp_matrix * Mat(v2_3d));

		// Well, in in principle, we'd have to do the whole stuff as in render(), like
		// clipping against the frustums etc.
		// But as long as our model is fully on the screen, we're fine.

		// Todo: This is all very similar to processProspectiveTri(...), except the other function does texturing stuff as well. Remove code duplication!

		//if (doBackfaceCulling) {
		if (!detail::are_vertices_ccw_in_screen_space(v0, v1, v2))
			continue;
		//}

		cv::Rect bbox = detail::calculate_clipped_bounding_box(v0, v1, v2, viewport_width, viewport_height);
		int minX = bbox.x;
		int maxX = bbox.x + bbox.width;
		int minY = bbox.y;
		int maxY = bbox.y + bbox.height;

		//if (t.maxX <= t.minX || t.maxY <= t.minY) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		//	continue;

		bool whole_triangle_is_visible = true;
		for (int yi = minY; yi <= maxY; yi++)
		{
			for (int xi = minX; xi <= maxX; xi++)
			{
				// we want centers of pixels to be used in computations. TODO: Do we?
				float x = (float)xi + 0.5f;
				float y = (float)yi + 0.5f;
				// these will be used for barycentric weights computation
				double one_over_v0ToLine12 = 1.0 / detail::implicit_line(v0[0], v0[1], v1, v2);
				double one_over_v1ToLine20 = 1.0 / detail::implicit_line(v1[0], v1[1], v2, v0);
				double one_over_v2ToLine01 = 1.0 / detail::implicit_line(v2[0], v2[1], v0, v1);
				// affine barycentric weights
				double alpha = detail::implicit_line(x, y, v1, v2) * one_over_v0ToLine12;
				double beta = detail::implicit_line(x, y, v2, v0) * one_over_v1ToLine20;
				double gamma = detail::implicit_line(x, y, v0, v1) * one_over_v2ToLine01;
				// if pixel (x, y) is inside the triangle or on one of its edges
				if (alpha >= 0 && beta >= 0 && gamma >= 0)
				{
					double z_affine = alpha*(double)v0[2] + beta*(double)v1[2] + gamma*(double)v2[2];
					// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
					if (z_affine < depth_buffer.at<double>(yi, xi)/* && z_affine <= 1.0*/) {
						whole_triangle_is_visible = false;
						break;
					}
					else {

					}
				}
			}
			if (!whole_triangle_is_visible) {
				break;
			}
		}

		if (!whole_triangle_is_visible) {
			continue;
		}

		cv::Point2f src_tri[3];
		cv::Point2f dst_tri[3];

		Vec4f vec(mesh.vertices[triangle_indices[0]][0], mesh.vertices[triangle_indices[0]][1], mesh.vertices[triangle_indices[0]][2], 1.0f);
		Vec4f res = Mat(affine_camera_matrix * Mat(vec));
		src_tri[0] = Vec2f(res[0], res[1]);

		vec = Vec4f(mesh.vertices[triangle_indices[1]][0], mesh.vertices[triangle_indices[1]][1], mesh.vertices[triangle_indices[1]][2], 1.0f);
		res = Mat(affine_camera_matrix * Mat(vec));
		src_tri[1] = Vec2f(res[0], res[1]);

		vec = Vec4f(mesh.vertices[triangle_indices[2]][0], mesh.vertices[triangle_indices[2]][1], mesh.vertices[triangle_indices[2]][2], 1.0f);
		res = Mat(affine_camera_matrix * Mat(vec));
		src_tri[2] = Vec2f(res[0], res[1]);

		dst_tri[0] = cv::Point2f(texture_map.cols*mesh.texcoords[triangle_indices[0]][0], texture_map.rows*mesh.texcoords[triangle_indices[0]][1] - 1.0f);
		dst_tri[1] = cv::Point2f(texture_map.cols*mesh.texcoords[triangle_indices[1]][0], texture_map.rows*mesh.texcoords[triangle_indices[1]][1] - 1.0f);
		dst_tri[2] = cv::Point2f(texture_map.cols*mesh.texcoords[triangle_indices[2]][0], texture_map.rows*mesh.texcoords[triangle_indices[2]][1] - 1.0f);

		// Get the inverse Affine Transform from original image: from dst to src
		Mat warp_mat_org_inv = cv::getAffineTransform(dst_tri, src_tri);
		warp_mat_org_inv.convertTo(warp_mat_org_inv, CV_32FC1);

		// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
		for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
			for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
				if (is_point_in_triangle(cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {
					if (mapping_type == TextureInterpolation::Area){

						//calculate positions of 4 corners of pixel in image (src)
						Vec3f homogenous_dst_upper_left = Vec3f(x - 0.5, y - 0.5, 1.f);
						Vec3f homogenous_dst_upper_right = Vec3f(x + 0.5, y - 0.5, 1.f);
						Vec3f homogenous_dst_lower_left = Vec3f(x - 0.5, y + 0.5, 1.f);
						Vec3f homogenous_dst_lower_right = Vec3f(x + 0.5, y + 0.5, 1.f);

						Vec2f src_texel_upper_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_left));
						Vec2f src_texel_upper_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_right));
						Vec2f src_texel_lower_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_left));
						Vec2f src_texel_lower_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_right));

						float min_a = min(min(src_texel_upper_left[0], src_texel_upper_right[0]), min(src_texel_lower_left[0], src_texel_lower_right[0]));
						float max_a = max(max(src_texel_upper_left[0], src_texel_upper_right[0]), max(src_texel_lower_left[0], src_texel_lower_right[0]));
						float min_b = min(min(src_texel_upper_left[1], src_texel_upper_right[1]), min(src_texel_lower_left[1], src_texel_lower_right[1]));
						float max_b = max(max(src_texel_upper_left[1], src_texel_upper_right[1]), max(src_texel_lower_left[1], src_texel_lower_right[1]));

						cv::Vec3i color = cv::Vec3i();
						int num_texels = 0;

						for (int a = ceil(min_a); a <= floor(max_a); ++a)
						{
							for (int b = ceil(min_b); b <= floor(max_b); ++b)
							{
								if (is_point_in_triangle(cv::Point2f(a, b), src_texel_upper_left, src_texel_lower_left, src_texel_upper_right) || is_point_in_triangle(cv::Point2f(a, b), src_texel_lower_left, src_texel_upper_right, src_texel_lower_right)) {
									if (a < image.cols && b < image.rows){ //if src_texel in triangle and in image
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
							Vec3f homogenous_dst_coord = Vec3f(x, y, 1.f);
							Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

							if ((cvRound(src_texel[1]) < image.rows) && cvRound(src_texel[0]) < image.cols) {
								color = image.at<cv::Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]));
							}
						}
						texture_map.at<cv::Vec3b>(y, x) = color;
					}
					else if (mapping_type == TextureInterpolation::Bilinear){

						//calculate corrresponding position of dst_coord pixel center in image (src)
						Vec3f homogenous_dst_coord = Vec3f(x, y, 1.f);
						Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						//calculate distances to next 4 pixels
						float distance_upper_left = sqrt(powf(src_texel[0] - floor(src_texel[0]), 2) + powf(src_texel[1] - floor(src_texel[1]), 2));
						float distance_upper_right = sqrt(powf(src_texel[0] - floor(src_texel[0]), 2) + powf(src_texel[1] - ceil(src_texel[1]), 2));
						float distance_lower_left = sqrt(powf(src_texel[0] - ceil(src_texel[0]), 2) + powf(src_texel[1] - floor(src_texel[1]), 2));
						float distance_lower_right = sqrt(powf(src_texel[0] - ceil(src_texel[0]), 2) + powf(src_texel[1] - ceil(src_texel[1]), 2));

						//normalize distances
						float sum_distances = distance_lower_left + distance_lower_right + distance_upper_left + distance_upper_right;
						distance_lower_left /= sum_distances;
						distance_lower_right /= sum_distances;
						distance_upper_left /= sum_distances;
						distance_upper_right /= sum_distances;

						// set color depending on distance from next 4 pixels
						for (int color = 0; color < 3; color++){
							float color_upper_left = image.at<cv::Vec3b>(floor(src_texel[1]), floor(src_texel[0]))[color] * distance_upper_left;
							float color_upper_right = image.at<cv::Vec3b>(floor(src_texel[1]), ceil(src_texel[0]))[color] * distance_upper_right;
							float color_lower_left = image.at<cv::Vec3b>(ceil(src_texel[1]), floor(src_texel[0]))[color] * distance_lower_left;
							float color_lower_right = image.at<cv::Vec3b>(ceil(src_texel[1]), ceil(src_texel[0]))[color] * distance_lower_right;

							texture_map.at<cv::Vec3b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
						}
					}
					else if (mapping_type == TextureInterpolation::NearestNeighbour){

						//calculate corrresponding position of dst_coord pixel center in image (src)
						Vec3f homogenous_dst_coord = Vec3f(x, y, 1.f);
						Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						if ((cvRound(src_texel[1]) < image.rows) && (cvRound(src_texel[0]) < image.cols))
							texture_map.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]));
					}
				}
			}
		}
	}
	return texture_map;
};

	} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_EXTRACTION_HPP_ */