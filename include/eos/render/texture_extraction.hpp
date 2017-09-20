/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/texture_extraction.hpp
 *
 * Copyright 2014-2017 Patrik Huber
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

#include "eos/core/Image.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/render/detail/texture_extraction_detail.hpp"
#include "eos/render/render_affine.hpp"
//#include "eos/render/utils.hpp" // for clip_to_screen_space() in v2::
//#include "eos/render/Rasterizer.hpp"
//#include "eos/render/FragmentShader.hpp"
#include "eos/fitting/closest_edge_fitting.hpp" // for ray_triangle_intersect(). Move to eos/render/raycasting.hpp?

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "Eigen/Core"
#include "Eigen/QR"

#include <tuple>
#include <cassert>
#include <future>
#include <vector>
#include <array>
#include <cstddef>
#include <cmath>

namespace eos {
	namespace render {

/* This function is copied from OpenCV,	originally under BSD licence.
 * imgwarp.cpp from OpenCV-3.2.0.
 *
 * Calculates coefficients of affine transformation
 * which maps (xi,yi) to (ui,vi), (i=1,2,3):
 *
 * ui = c00*xi + c01*yi + c02
 *
 * vi = c10*xi + c11*yi + c12
 *
 * Coefficients are calculated by solving linear system:
 * / x0 y0  1  0  0  0 \ /c00\ /u0\
 * | x1 y1  1  0  0  0 | |c01| |u1|
 * | x2 y2  1  0  0  0 | |c02| |u2|
 * |  0  0  0 x0 y0  1 | |c10| |v0|
 * |  0  0  0 x1 y1  1 | |c11| |v1|
 * \  0  0  0 x2 y2  1 / |c12| |v2|
 *
 * where:
 *   cij - matrix coefficients
 */
// Note: The original functions used doubles.
Eigen::Matrix<float, 2, 3> get_affine_transform(const std::array<Eigen::Vector2f, 3>& src, const std::array<Eigen::Vector2f, 3>& dst)
{
	using Eigen::Matrix;
	assert(src.size() == dst.size() && src.size() == 3);
	
	Matrix<float, 6, 6> A;
	Matrix<float, 6, 1> b;

    for( int i = 0; i < 3; i++ )
    {
		A.block<1, 2>(2 * i, 0) = src[i]; // the odd rows
		A.block<1, 2>((2 * i) + 1, 3) = src[i]; // even rows
		A(2 * i, 2) = 1.0f;
		A((2 * i) + 1, 5) = 1.0f;
		A.block<1, 3>(2 * i, 3).setZero();
		A.block<1, 3>((2 * i) + 1, 0).setZero();
		b.segment<2>(2 * i) = dst[i];
    }

	Matrix<float, 6, 1> X = A.colPivHouseholderQr().solve(b);

	Matrix<float, 2, 3> transform_matrix;
	transform_matrix.block<1, 3>(0, 0) = X.segment<3>(0);
	transform_matrix.block<1, 3>(1, 0) = X.segment<3>(3);

	return transform_matrix;
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

// Forward declarations:
core::Image4u extract_texture(const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix, const core::Image3u& image, const core::Image1d& depthbuffer, bool compute_view_angle, TextureInterpolation mapping_type, int isomap_resolution);
namespace detail { core::Image4u interpolate_black_line(core::Image4u& isomap); }

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 *
 * Note/Todo: Only use TextureInterpolation::NearestNeighbour
 * for the moment, the other methods don't have correct handling of
 * the alpha channel (and will most likely throw an exception).
 *
 * Todo: These should be renamed to extract_texture_affine? Can we combine both cases somehow?
 * Or an overload with RenderingParameters?
 *
 * For TextureInterpolation::NearestNeighbour, returns a 4-channel isomap
 * with the visibility in the 4th channel (0=invis, 255=visible).
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from. Should be 8UC3, other types not supported yet.
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned. If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90°, 127 meaning facing a 45° angle and 255 meaning front-facing, and all values in between). If set to false, the alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline core::Image4u extract_texture(const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix, const core::Image3u& image, bool compute_view_angle = false, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	// Render the model to get a depth buffer:
	core::Image1d depthbuffer;
	std::tie(std::ignore, depthbuffer) = render::render_affine(mesh, affine_camera_matrix, image.cols, image.rows);
	// Note: There's potential for optimisation here - we don't need to do everything that is done in render_affine to just get the depthbuffer.

	// Now forward the call to the actual texture extraction function:
	return extract_texture(mesh, affine_camera_matrix, image, depthbuffer, compute_view_angle, mapping_type, isomap_resolution);
};

/**
 * Extracts the texture of the face from the given image
 * and stores it as isomap (a rectangular texture map).
 * This function can be used if a depth buffer has already been computed.
 * To just run the texture extraction, see the overload
 * extract_texture(Mesh, cv::Mat, cv::Mat, TextureInterpolation, int). // Todo: I think this signature needs updating.
 *
 * It might be wise to remove this overload as it can get quite confusing
 * with the zbuffer. Obviously the depthbuffer given should have been created
 * with the same (affine or ortho) projection matrix than the texture extraction is called with.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] affine_camera_matrix An estimated 3x4 affine camera matrix.
 * @param[in] image The image to extract the texture from.
 * @param[in] depthbuffer A pre-calculated depthbuffer image.
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned. If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90°, 127 meaning facing a 45° angle and 255 meaning front-facing, and all values in between). If set to false, the alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline core::Image4u extract_texture(const core::Mesh& mesh, Eigen::Matrix<float, 3, 4> affine_camera_matrix, const core::Image3u& image, const core::Image1d& depthbuffer, bool compute_view_angle = false, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	assert(mesh.vertices.size() == mesh.texcoords.size());

	using Eigen::Vector2f;
	using Eigen::Vector3f;
	using Eigen::Vector4f;
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;
	using std::round;
	using std::sqrt;
	using std::pow;

	Eigen::Matrix<float, 4, 4> affine_camera_matrix_with_z = detail::calculate_affine_z_direction(affine_camera_matrix);

	// Todo: We should handle gray images, but output a 4-channel isomap nevertheless I think.
	core::Image4u isomap(isomap_resolution, isomap_resolution); // We should initialise with zeros. Incidentially, the current Image4u c'tor does that.

	std::vector<std::future<void>> results;
	for (const auto& triangle_indices : mesh.tvi) {

		// Note: If there's a performance problem, there's no need to capture the whole mesh - we could capture only the three required vertices with their texcoords.
		auto extract_triangle = [&mesh, &affine_camera_matrix_with_z, &triangle_indices, &depthbuffer, &isomap, &mapping_type, &image, &compute_view_angle]() {

			// Find out if the current triangle is visible:
			// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
			// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
			// the texture.
			// Possible improvement: - If only part of the triangle is visible, split it

			// This could be optimized in 2 ways though:
			// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
			// - We transform them later (below) a second time. Only do it once.

			const Vector4f v0_as_Vector4f(mesh.vertices[triangle_indices[0]][0], mesh.vertices[triangle_indices[0]][1], mesh.vertices[triangle_indices[0]][2], 1.0f);
                        const Vector4f v1_as_Vector4f(mesh.vertices[triangle_indices[1]][0], mesh.vertices[triangle_indices[1]][1], mesh.vertices[triangle_indices[1]][2], 1.0f);
                        const Vector4f v2_as_Vector4f(mesh.vertices[triangle_indices[2]][0], mesh.vertices[triangle_indices[2]][1], mesh.vertices[triangle_indices[2]][2], 1.0f);

			// Project the triangle vertices to screen coordinates, and use the depthbuffer to check whether the triangle is visible:
			const Vector4f v0 = affine_camera_matrix_with_z * v0_as_Vector4f;
			const Vector4f v1 = affine_camera_matrix_with_z * v1_as_Vector4f;
			const Vector4f v2 = affine_camera_matrix_with_z * v2_as_Vector4f;

			if (!detail::is_triangle_visible(glm::tvec4<float>(v0[0], v0[1], v0[2], v0[3]), glm::tvec4<float>(v1[0], v1[1], v1[2], v1[3]), glm::tvec4<float>(v2[0], v2[1], v2[2], v2[3]), depthbuffer))
			{
				//continue;
				return;
			}

			float alpha_value;
			if (compute_view_angle)
			{
				// Calculate how well visible the current triangle is:
				// (in essence, the dot product of the viewing direction (0, 0, 1) and the face normal)
				const Vector3f face_normal = compute_face_normal(v0_as_Vector4f, v1_as_Vector4f, v2_as_Vector4f);
				// Transform the normal to "screen" (kind of "eye") space using the upper 3x3 part of the affine camera matrix (=the translation can be ignored):
				Vector3f face_normal_transformed = affine_camera_matrix_with_z.block<3, 3>(0, 0) * face_normal;
				face_normal_transformed.normalize(); // normalise to unit length
				// Implementation notes regarding the affine camera matrix and the sign:
				// If the matrix given were the model_view matrix, the sign would be correct.
				// However, affine_camera_matrix includes glm::ortho, which includes a z-flip.
				// So we need to flip one of the two signs.
				// * viewing_direction(0.0f, 0.0f, 1.0f) is correct if affine_camera_matrix were only a model_view matrix
				// * affine_camera_matrix includes glm::ortho, which flips z, so we flip the sign of viewing_direction.
				// We don't need the dot product since viewing_direction.xy are 0 and .z is 1:
				const float angle = -face_normal_transformed[2]; // flip sign, see above
				assert(angle >= -1.f && angle <= 1.f);
				// angle is [-1, 1].
				//  * +1 means   0° (same direction)
				//  *  0 means  90°
				//  * -1 means 180° (facing opposite directions)
				// It's a linear relation, so +0.5 is 45° etc.
				// An angle larger than 90° means the vertex won't be rendered anyway (because it's back-facing) so we encode 0° to 90°.
				if (angle < 0.0f) {
					alpha_value = 0.0f;
				} else {
					alpha_value = angle * 255.0f;
				}
			}
			else {
				// no visibility angle computation - if the triangle/pixel is visible, set the alpha chan to 255 (fully visible pixel).
				alpha_value = 255.0f;
			}

			// Todo: Documentation
			std::array<Vector2f, 3> src_tri;
			std::array<Vector2f, 3> dst_tri;

			Vector4f vec(mesh.vertices[triangle_indices[0]][0], mesh.vertices[triangle_indices[0]][1], mesh.vertices[triangle_indices[0]][2], 1.0f);
			Vector4f res = affine_camera_matrix_with_z * vec;
			src_tri[0] = Vector2f(res[0], res[1]);

			vec = Vector4f(mesh.vertices[triangle_indices[1]][0], mesh.vertices[triangle_indices[1]][1], mesh.vertices[triangle_indices[1]][2], 1.0f);
			res = affine_camera_matrix_with_z * vec;
			src_tri[1] = Vector2f(res[0], res[1]);

			vec = Vector4f(mesh.vertices[triangle_indices[2]][0], mesh.vertices[triangle_indices[2]][1], mesh.vertices[triangle_indices[2]][2], 1.0f);
			res = affine_camera_matrix_with_z * vec;
			src_tri[2] = Vector2f(res[0], res[1]);

			dst_tri[0] = Vector2f((isomap.cols - 0.5)*mesh.texcoords[triangle_indices[0]][0], (isomap.rows - 0.5)*mesh.texcoords[triangle_indices[0]][1]);
			dst_tri[1] = Vector2f((isomap.cols - 0.5)*mesh.texcoords[triangle_indices[1]][0], (isomap.rows - 0.5)*mesh.texcoords[triangle_indices[1]][1]);
			dst_tri[2] = Vector2f((isomap.cols - 0.5)*mesh.texcoords[triangle_indices[2]][0], (isomap.rows - 0.5)*mesh.texcoords[triangle_indices[2]][1]);

			// We now have the source triangles in the image and the source triangle in the isomap
			// We use the inverse/ backward mapping approach, so we want to find the corresponding texel (texture-pixel) for each pixel in the isomap

			// Get the inverse Affine Transform from original image: from dst (pixel in isomap) to src (in image)
			Eigen::Matrix<float, 2, 3> warp_mat_org_inv = get_affine_transform(dst_tri, src_tri);

			// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
			for (int x = min(dst_tri[0][0], min(dst_tri[1][0], dst_tri[2][0])); x < max(dst_tri[0][0], max(dst_tri[1][0], dst_tri[2][0])); ++x) {
				for (int y = min(dst_tri[0][1], min(dst_tri[1][1], dst_tri[2][1])); y < max(dst_tri[0][1], max(dst_tri[1][1], dst_tri[2][1])); ++y) {
					if (detail::is_point_in_triangle(Vector2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {

						// As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to
						// calculate the pixel colors depending on the next texels
						// there are three different texture interpolation methods: area, bilinear and nearest neighbour

						// Area mapping: calculate mean color of texels in transformed pixel area
						if (mapping_type == TextureInterpolation::Area) {

							// calculate positions of 4 corners of pixel in image (src)
							const Vector3f homogenous_dst_upper_left(x - 0.5f, y - 0.5f, 1.0f);
							const Vector3f homogenous_dst_upper_right(x + 0.5f, y - 0.5f, 1.0f);
							const Vector3f homogenous_dst_lower_left(x - 0.5f, y + 0.5f, 1.0f);
							const Vector3f homogenous_dst_lower_right(x + 0.5f, y + 0.5f, 1.0f);

							const Vector2f src_texel_upper_left = warp_mat_org_inv * homogenous_dst_upper_left;
							const Vector2f src_texel_upper_right = warp_mat_org_inv * homogenous_dst_upper_right;
							const Vector2f src_texel_lower_left = warp_mat_org_inv * homogenous_dst_lower_left;
							const Vector2f src_texel_lower_right = warp_mat_org_inv * homogenous_dst_lower_right;

							float min_a = min(min(src_texel_upper_left[0], src_texel_upper_right[0]), min(src_texel_lower_left[0], src_texel_lower_right[0]));
							float max_a = max(max(src_texel_upper_left[0], src_texel_upper_right[0]), max(src_texel_lower_left[0], src_texel_lower_right[0]));
							float min_b = min(min(src_texel_upper_left[1], src_texel_upper_right[1]), min(src_texel_lower_left[1], src_texel_lower_right[1]));
							float max_b = max(max(src_texel_upper_left[1], src_texel_upper_right[1]), max(src_texel_lower_left[1], src_texel_lower_right[1]));

							Eigen::Vector3i color; // std::uint8_t actually.
							int num_texels = 0;

							// loop over square in which quadrangle out of the four corners of pixel is
							for (int a = ceil(min_a); a <= floor(max_a); ++a)
							{
								for (int b = ceil(min_b); b <= floor(max_b); ++b)
								{
									// check if texel is in quadrangle
									if (detail::is_point_in_triangle(Vector2f(a, b), src_texel_upper_left, src_texel_lower_left, src_texel_upper_right) || detail::is_point_in_triangle(Vector2f(a, b), src_texel_lower_left, src_texel_upper_right, src_texel_lower_right)) {
										if (a < image.cols && b < image.rows) { // check if texel is in image
											num_texels++;
											color += Eigen::Vector3i(image(b, a)[0], image(b, a)[1], image(b, a)[2]);
										}
									}
								}
							}
							if (num_texels > 0)
								color = color / num_texels;
							else { // if no corresponding texel found, nearest neighbour interpolation
								// calculate corresponding position of dst_coord pixel center in image (src)
								Vector3f homogenous_dst_coord(x, y, 1.0f);
								Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

								if ((round(src_texel[1]) < image.rows) && round(src_texel[0]) < image.cols) {
									const int y = round(src_texel[1]);
									const int x = round(src_texel[0]);
									color = Eigen::Vector3i(image(y, x)[0], image(y, x)[1], image(y, x)[2]);
								}
							}
							isomap(y, x) = { static_cast<std::uint8_t>(color[0]), static_cast<std::uint8_t>(color[1]), static_cast<std::uint8_t>(color[2]), static_cast<std::uint8_t>(alpha_value) };
						}
						// Bilinear mapping: calculate pixel color depending on the four neighbouring texels
						else if (mapping_type == TextureInterpolation::Bilinear) {

							// calculate corresponding position of dst_coord pixel center in image (src)
							const Vector3f homogenous_dst_coord(x, y, 1.0f);
							const Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

							// calculate euclidean distances to next 4 texels
							float distance_upper_left = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
							float distance_upper_right = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));
							float distance_lower_left = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
							float distance_lower_right = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));

							// normalise distances that the sum of all distances is 1
							const float sum_distances = distance_lower_left + distance_lower_right + distance_upper_left + distance_upper_right;
							distance_lower_left /= sum_distances;
							distance_lower_right /= sum_distances;
							distance_upper_left /= sum_distances;
							distance_upper_right /= sum_distances;

							// set color depending on distance from next 4 texels
							// (we map the data from std::array<uint8_t, 3> to an Eigen::Map, then cast that to float to multiply with the float-scalar distance.)
							// (this is untested!)
							const Vector3f color_upper_left = Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(image(floor(src_texel[1]), floor(src_texel[0])).data(), 3).cast<float>() * distance_upper_left;
							const Vector3f color_upper_right = Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(image(floor(src_texel[1]), ceil(src_texel[0])).data(), 3).cast<float>() * distance_upper_right;
							const Vector3f color_lower_left = Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(image(ceil(src_texel[1]), floor(src_texel[0])).data(), 3).cast<float>() * distance_lower_left;
							const Vector3f color_lower_right = Eigen::Map<const Eigen::Matrix<std::uint8_t, 1, 3>>(image(ceil(src_texel[1]), ceil(src_texel[0])).data(), 3).cast<float>() * distance_lower_right;

							//isomap(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
							isomap(y, x)[0] = static_cast<std::uint8_t>(glm::clamp(color_upper_left[0] + color_upper_right[0] + color_lower_left[0] + color_lower_right[0], 0.f, 255.0f));
							isomap(y, x)[1] = static_cast<std::uint8_t>(glm::clamp(color_upper_left[1] + color_upper_right[1] + color_lower_left[1] + color_lower_right[1], 0.f, 255.0f));
							isomap(y, x)[2] = static_cast<std::uint8_t>(glm::clamp(color_upper_left[2] + color_upper_right[2] + color_lower_left[2] + color_lower_right[2], 0.f, 255.0f));
							isomap(y, x)[3] = static_cast<std::uint8_t>(alpha_value); // pixel is visible
						}
						// NearestNeighbour mapping: set color of pixel to color of nearest texel
						else if (mapping_type == TextureInterpolation::NearestNeighbour) {

							// calculate corresponding position of dst_coord pixel center in image (src)
							const Vector3f homogenous_dst_coord(x, y, 1.0f);
							const Vector2f src_texel = warp_mat_org_inv * homogenous_dst_coord;

							if ((round(src_texel[1]) < image.rows) && (round(src_texel[0]) < image.cols) && round(src_texel[0]) > 0 && round(src_texel[1]) > 0)
							{
								isomap(y, x)[0] = image(round(src_texel[1]), round(src_texel[0]))[0];
								isomap(y, x)[1] = image(round(src_texel[1]), round(src_texel[0]))[1];
								isomap(y, x)[2] = image(round(src_texel[1]), round(src_texel[0]))[2];
								isomap(y, x)[3] = static_cast<std::uint8_t>(alpha_value); // pixel is visible
							}
						}
					}
				}
			}
		}; // end lambda auto extract_triangle();
		results.emplace_back(std::async(extract_triangle));
	} // end for all mesh.tvi
	// Collect all the launched tasks:
	for (auto&& r : results) {
		r.get();
	}

	// Workaround for the black line in the isomap (see GitHub issue #4):
/*	if (mesh.texcoords.size() <= 3448)
	{
		isomap = detail::interpolate_black_line(isomap);
	}
*/
	return isomap;
};

/* New texture extraction, will replace above one at some point: */
namespace v2 {

/**
 * @brief Extracts the texture of the face from the given image and stores it as isomap (a rectangular texture map).
 *
 * New texture extraction, will replace above one at some point.
 * Copy the documentation from above extract_texture function, once we replace it.
 *
 * Note/Todo: Add an overload that takes a vector of bool / visible vertices, for the case when we already computed the visibility? (e.g. for edge-fitting)
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] view_model_matrix Todo.
 * @param[in] projection_matrix Todo.
 * @param[in] viewport Not needed at the moment. Might be, if we change clip_to_screen_space() to take a viewport.
 * @param[in] image The image to extract the texture from. Todo: Does it have to be 8UC3 or something, or does it not matter?
 * @param[in] compute_view_angle Unused at the moment.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
//cv::Mat extract_texture(const core::Mesh& mesh, glm::mat4x4 view_model_matrix, glm::mat4x4 projection_matrix,
//                        glm::vec4 /*viewport, not needed at the moment */, cv::Mat image,
//                        bool /* compute_view_angle, unused atm */, int isomap_resolution = 512)
/*
{
    using detail::divide_by_w;
    using glm::vec2;
    using glm::vec3;
    using glm::vec4;
    using std::vector;
    // actually we only need a rasteriser for this!
    Rasterizer<ExtractionFragmentShader> extraction_rasterizer(isomap_resolution, isomap_resolution);
    Texture image_to_extract_from_as_tex = create_mipmapped_texture(image, 1);
    extraction_rasterizer.enable_depth_test = false;
    extraction_rasterizer.extracting_tex = true;

    vector<bool> visibility_ray;
    vector<vec4> rotated_vertices;
    // In perspective case... does the perspective projection matrix not change visibility? Do we not need to
    // apply it?
    // (If so, then we can change the two input matrices to this function to one (mvp_matrix)).
    std::for_each(std::begin(mesh.vertices), std::end(mesh.vertices),
                  [&rotated_vertices, &view_model_matrix](auto&& v) {
                      rotated_vertices.push_back(view_model_matrix * v);
                  });
    // This code is duplicated from the edge-fitting. I think I can put this into a function in the library.
    for (const auto& vertex : rotated_vertices)
    {
        bool visible = true;
        // For every tri of the rotated mesh:
        for (auto&& tri : mesh.tvi)
        {
            auto& v0 = rotated_vertices[tri[0]]; // const?
            auto& v1 = rotated_vertices[tri[1]];
            auto& v2 = rotated_vertices[tri[2]];

            vec3 ray_origin(vertex);
            vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
            auto intersect = fitting::ray_triangle_intersect(ray_origin, ray_direction, vec3(v0), vec3(v1),
                                                             vec3(v2), false);
            // first is bool intersect, second is the distance t
            if (intersect.first == true)
            {
                // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the
                // intersection:
                // Check if in front or behind?
                if (intersect.second.get() <= 1e-4)
                {
                    continue; // the intersection is behind the vertex, we don't care about it
                }
                // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
                visible = false;
                break;
            }
        }
        visibility_ray.push_back(visible);
    }

    vector<vec4> wnd_coords; // will contain [x_wnd, y_wnd, z_ndc, 1/w_clip]
    for (auto&& vtx : mesh.vertices)
    {
        auto clip_coords = projection_matrix * view_model_matrix * vtx;
        clip_coords = divide_by_w(clip_coords);
        const vec2 screen_coords = clip_to_screen_space(clip_coords.x, clip_coords.y, image.cols, image.rows);
        clip_coords.x = screen_coords.x;
        clip_coords.y = screen_coords.y;
        wnd_coords.push_back(clip_coords);
    }

    // Go on with extracting: This only needs the rasteriser/FS, not the whole Renderer.
    const int tex_width = isomap_resolution;
    const int tex_height =
        isomap_resolution; // keeping this in case we need non-square texture maps at some point
    for (const auto& tvi : mesh.tvi)
    {
        if (visibility_ray[tvi[0]] && visibility_ray[tvi[1]] &&
            visibility_ray[tvi[2]]) // can also try using ||, but...
        {
            // Test with a rendered & re-extracted texture shows that we're off by a pixel or more,
            // definitely need to correct this. Probably here.
            // It looks like it is 1-2 pixels off. Definitely a bit more than 1.
            detail::Vertex<double> pa{
                vec4(mesh.texcoords[tvi[0]][0] * tex_width,
					 mesh.texcoords[tvi[0]][1] * tex_height,
                     wnd_coords[tvi[0]].z, // z_ndc
					 wnd_coords[tvi[0]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[0]].x / image.cols,
                    wnd_coords[tvi[0]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            detail::Vertex<double> pb{
                vec4(mesh.texcoords[tvi[1]][0] * tex_width,
				mesh.texcoords[tvi[1]][1] * tex_height,
                wnd_coords[tvi[1]].z, // z_ndc
				wnd_coords[tvi[1]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[1]].x / image.cols,
                    wnd_coords[tvi[1]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            detail::Vertex<double> pc{
                vec4(mesh.texcoords[tvi[2]][0] * tex_width,
				mesh.texcoords[tvi[2]][1] * tex_height,
                wnd_coords[tvi[2]].z, // z_ndc 
				wnd_coords[tvi[2]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[2]].x / image.cols,
                    wnd_coords[tvi[2]].y / image.rows // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            extraction_rasterizer.raster_triangle(pa, pb, pc, image_to_extract_from_as_tex);
        }
    }

    return extraction_rasterizer.colorbuffer;
};
*/

} /* namespace v2 */

namespace detail {

// Workaround for the pixels that don't get filled in extract_texture().
// There's a vertical line of missing values in the middle of the isomap,
// as well as a few pixels on a horizontal line around the mouth. They
// manifest themselves as black lines in the final isomap. This function
// just fills these missing values by interpolating between two neighbouring
// pixels. See GitHub issue #4.
inline core::Image4u interpolate_black_line(core::Image4u& isomap)
{
	// Replace the vertical black line ("missing data"):
	using RGBAType = Eigen::Matrix<std::uint8_t, 1, 4>;
	using Eigen::Map;
        const int col = isomap.cols / 2;
	for (int row = 0; row < isomap.rows; ++row)
	{
		if (isomap(row, col) == std::array<std::uint8_t, 4>{ 0, 0, 0, 0 })
		{
			Eigen::Vector4f pixel_val = Map<const RGBAType>(isomap(row, col - 1).data(), 4).cast<float>() * 0.5f + Map<const RGBAType>(isomap(row, col + 1).data(), 4).cast<float>() * 0.5f;
			isomap(row, col) = { static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]), static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3]) };
			
		}
	}

	// Replace the horizontal line around the mouth that occurs in the
	// isomaps of resolution 512x512 and higher:
	if (isomap.rows == 512) // num cols is 512 as well
	{
                const int r = 362;
		for (int c = 206; c <= 306; ++c)
		{
			if (isomap(r, c) == std::array<std::uint8_t, 4>{ 0, 0, 0, 0 })
			{
				Eigen::Vector4f pixel_val = Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f + Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
				isomap(r, c) = { static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]), static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3]) };
			}
		}
	}
	if (isomap.rows == 1024) // num cols is 1024 as well
	{
		int r = 724;
		for (int c = 437; c <= 587; ++c)
		{
			if (isomap(r, c) == std::array<std::uint8_t, 4>{ 0, 0, 0, 0 })
			{
				Eigen::Vector4f pixel_val = Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f + Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
				isomap(r, c) = { static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]), static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3]) };
			}
		}
		r = 725;
		for (int c = 411; c <= 613; ++c)
		{
			if (isomap(r, c) == std::array<std::uint8_t, 4>{ 0, 0, 0, 0 })
			{
				Eigen::Vector4f pixel_val = Map<const RGBAType>(isomap(r - 1, c).data(), 4).cast<float>() * 0.5f + Map<const RGBAType>(isomap(r + 1, c).data(), 4).cast<float>() * 0.5f;
				isomap(r, c) = { static_cast<std::uint8_t>(pixel_val[0]), static_cast<std::uint8_t>(pixel_val[1]), static_cast<std::uint8_t>(pixel_val[2]), static_cast<std::uint8_t>(pixel_val[3]) };
			}
		}
	}
	// Higher resolutions are probably affected as well but not used so far in practice.

	return isomap;
};
} /* namespace detail */

	} /* namespace render */
} /* namespace eos */

#endif /* TEXTURE_EXTRACTION_HPP_ */
