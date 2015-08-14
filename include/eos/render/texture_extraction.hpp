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

#include "eos/render/detail/texture_extraction_detail.hpp"
#include "eos/render/Mesh.hpp"
#include "eos/render/render_affine.hpp"
#include "eos/render/detail/render_detail.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <tuple>
#include <cassert>

namespace eos {
	namespace render {

/**
 * The interpolation types that can be used to map the
 * texture from the original image to the isomap.
 */
enum class TextureInterpolation {
	NearestNeighbour,
	Bilinear,
	Area
};

// Just a forward declaration
inline cv::Mat extract_texture(Mesh mesh, cv::Mat affine_camera_matrix, cv::Mat image, cv::Mat depthbuffer, TextureInterpolation mapping_type, int isomap_resolution);

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 *
 * Todo: These should be renamed to extract_texture_affine? Can we combine both cases somehow?
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline cv::Mat extract_texture(Mesh mesh, cv::Mat affine_camera_matrix, cv::Mat image, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	// Render the model to get a depth buffer:
	cv::Mat depthbuffer;
	std::tie(std::ignore, depthbuffer) = render::render_affine(mesh, affine_camera_matrix, image.cols, image.rows);
	// Note: There's potential for optimisation here - we don't need to do everything that is done in render_affine to just get the depthbuffer.

	// Now forward the call to the actual texture extraction function:
	return extract_texture(mesh, affine_camera_matrix, image, depthbuffer, mapping_type, isomap_resolution);
};

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 * This function can be used if a depth buffer has already been computed.
 * To just run the texture extraction, see the overload
 * extract_texture(Mesh, cv::Mat, cv::Mat, TextureInterpolation, int).
 *
 * It might be wise to remove this overload as it can get quite confusing
 * with the zbuffer. Obviously the depthbuffer given should have been created
 * with the same (affine or ortho) projection matrix than the texture extraction is called with.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from.
 * @param[in] depthbuffer A pre-calculated depthbuffer image.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline cv::Mat extract_texture(Mesh mesh, cv::Mat affine_camera_matrix, cv::Mat image, cv::Mat depthbuffer, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	assert(mesh.vertices.size() == mesh.texcoords.size());

	using cv::Mat;
	using cv::Vec2f;
	using cv::Vec3f;
	using cv::Vec4f;
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;

	affine_camera_matrix = detail::calculate_affine_z_direction(affine_camera_matrix);

	Mat isomap = Mat::zeros(isomap_resolution, isomap_resolution, CV_8UC3); // #Todo: We do want an alpha channel. Will be added soon-ish.
	// #Todo: We should handle gray images, but output a 3-channel isomap nevertheless I think.

	for (const auto& triangle_indices : mesh.tvi) {
		// Find out if the current triangle is visible:
		// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
		// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
		// the texture.
		// Possible improvement: - If only part of the triangle is visible, split it

		// This could be optimized in 2 ways though:
		// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
		// - We transform them later (below) a second time. Only do it once.

		// Project the triangle vertices to screen coordinates, and use the depthbuffer to check whether the triangle is visible:
		Vec4f v0 = Mat(affine_camera_matrix * Mat(mesh.vertices[triangle_indices[0]]));
		Vec4f v1 = Mat(affine_camera_matrix * Mat(mesh.vertices[triangle_indices[1]]));
		Vec4f v2 = Mat(affine_camera_matrix * Mat(mesh.vertices[triangle_indices[2]]));

		if (!detail::is_triangle_visible(v0, v1, v2, depthbuffer))
		{
			continue;
		}

		// Todo: Documentation
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

		dst_tri[0] = cv::Point2f(isomap.cols*mesh.texcoords[triangle_indices[0]][0], isomap.rows*mesh.texcoords[triangle_indices[0]][1] - 1.0f);
		dst_tri[1] = cv::Point2f(isomap.cols*mesh.texcoords[triangle_indices[1]][0], isomap.rows*mesh.texcoords[triangle_indices[1]][1] - 1.0f);
		dst_tri[2] = cv::Point2f(isomap.cols*mesh.texcoords[triangle_indices[2]][0], isomap.rows*mesh.texcoords[triangle_indices[2]][1] - 1.0f);

		// Get the inverse Affine Transform from original image: from dst to src
		Mat warp_mat_org_inv = cv::getAffineTransform(dst_tri, src_tri);
		warp_mat_org_inv.convertTo(warp_mat_org_inv, CV_32FC1);

		// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
		for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
			for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
				if (detail::is_point_in_triangle(cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {
					if (mapping_type == TextureInterpolation::Area){

						// calculate positions of 4 corners of pixel in image (src)
						Vec3f homogenous_dst_upper_left(x - 0.5, y - 0.5, 1.f);
						Vec3f homogenous_dst_upper_right(x + 0.5, y - 0.5, 1.f);
						Vec3f homogenous_dst_lower_left(x - 0.5, y + 0.5, 1.f);
						Vec3f homogenous_dst_lower_right(x + 0.5, y + 0.5, 1.f);

						Vec2f src_texel_upper_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_left));
						Vec2f src_texel_upper_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_upper_right));
						Vec2f src_texel_lower_left = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_left));
						Vec2f src_texel_lower_right = Mat(warp_mat_org_inv * Mat(homogenous_dst_lower_right));

						float min_a = min(min(src_texel_upper_left[0], src_texel_upper_right[0]), min(src_texel_lower_left[0], src_texel_lower_right[0]));
						float max_a = max(max(src_texel_upper_left[0], src_texel_upper_right[0]), max(src_texel_lower_left[0], src_texel_lower_right[0]));
						float min_b = min(min(src_texel_upper_left[1], src_texel_upper_right[1]), min(src_texel_lower_left[1], src_texel_lower_right[1]));
						float max_b = max(max(src_texel_upper_left[1], src_texel_upper_right[1]), max(src_texel_lower_left[1], src_texel_lower_right[1]));

						cv::Vec3i color;
						int num_texels = 0;

						for (int a = ceil(min_a); a <= floor(max_a); ++a)
						{
							for (int b = ceil(min_b); b <= floor(max_b); ++b)
							{
								if (detail::is_point_in_triangle(cv::Point2f(a, b), src_texel_upper_left, src_texel_lower_left, src_texel_upper_right) || detail::is_point_in_triangle(cv::Point2f(a, b), src_texel_lower_left, src_texel_upper_right, src_texel_lower_right)) {
									if (a < image.cols && b < image.rows) { // if src_texel in triangle and in image
										num_texels++;
										color += image.at<cv::Vec3b>(b, a);
									}
								}
							}
						}
						if (num_texels > 0)
							color = color / num_texels;
						else { // if no corresponding texel found, nearest neighbor interpolation
							// calculate corresponding position of dst_coord pixel center in image (src)
							Vec3f homogenous_dst_coord = Vec3f(x, y, 1.f);
							Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

							if ((cvRound(src_texel[1]) < image.rows) && cvRound(src_texel[0]) < image.cols) {
								color = image.at<cv::Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]));
							}
						}
						isomap.at<cv::Vec3b>(y, x) = color;
					}
					else if (mapping_type == TextureInterpolation::Bilinear) {

						// calculate corresponding position of dst_coord pixel center in image (src)
						Vec3f homogenous_dst_coord(x, y, 1.f);
						Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						// calculate distances to next 4 pixels
						float distance_upper_left = sqrt(powf(src_texel[0] - floor(src_texel[0]), 2) + powf(src_texel[1] - floor(src_texel[1]), 2));
						float distance_upper_right = sqrt(powf(src_texel[0] - floor(src_texel[0]), 2) + powf(src_texel[1] - ceil(src_texel[1]), 2));
						float distance_lower_left = sqrt(powf(src_texel[0] - ceil(src_texel[0]), 2) + powf(src_texel[1] - floor(src_texel[1]), 2));
						float distance_lower_right = sqrt(powf(src_texel[0] - ceil(src_texel[0]), 2) + powf(src_texel[1] - ceil(src_texel[1]), 2));

						// normalise distances
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

							isomap.at<cv::Vec3b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
						}
					}
					else if (mapping_type == TextureInterpolation::NearestNeighbour) {

						// calculate corresponding position of dst_coord pixel center in image (src)
						Vec3f homogenous_dst_coord = Vec3f(x, y, 1.f);
						Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

						if ((cvRound(src_texel[1]) < image.rows) && (cvRound(src_texel[0]) < image.cols))
							isomap.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]));
					}
				}
			}
		}
	}
	return isomap;
};

	} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_EXTRACTION_HPP_ */
