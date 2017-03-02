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

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/texture_extraction_detail.hpp"
#include "eos/render/render_affine.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/render/utils.hpp" // for clip_to_screen_space()
#include "eos/render/Rasterizer.hpp"
#include "eos/render/FragmentShader.hpp"
#include "eos/fitting/closest_edge_fitting.hpp" // for ray_triangle_intersect()

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <tuple>
#include <cassert>
#include <future>
#include <vector>

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

// Forward declarations:
cv::Mat extract_texture(core::Mesh mesh, cv::Mat affine_camera_matrix, cv::Mat image, cv::Mat depthbuffer, bool compute_view_angle, TextureInterpolation mapping_type, int isomap_resolution);
namespace detail { cv::Mat interpolate_black_line(cv::Mat isomap); }

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
inline cv::Mat extract_texture(const core::Mesh& mesh, cv::Mat affine_camera_matrix, cv::Mat image, bool compute_view_angle = false, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	// Render the model to get a depth buffer:
	cv::Mat depthbuffer;
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
 * @param[in] compute_view_angle A flag whether the view angle of each vertex should be computed and returned. If set to true, the angle will be encoded into the alpha channel (0 meaning occluded or facing away 90°, 127 meaning facing a 45° angle and 255 meaning front-facing, and all values in between). If set to false, the alpha channel will only contain 0 for occluded vertices and 255 for visible vertices.
 * @param[in] mapping_type The interpolation type to be used for the extraction.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline cv::Mat extract_texture(core::Mesh mesh, cv::Mat affine_camera_matrix, cv::Mat image, cv::Mat depthbuffer, bool compute_view_angle = false, TextureInterpolation mapping_type = TextureInterpolation::NearestNeighbour, int isomap_resolution = 512)
{
	assert(mesh.vertices.size() == mesh.texcoords.size());
	assert(image.type() == CV_8UC3); // the other cases are not yet supported

	using cv::Mat;
	using cv::Vec2f;
	using cv::Vec3f;
	using cv::Vec4f;
	using cv::Vec3b;
	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;

	affine_camera_matrix = detail::calculate_affine_z_direction(affine_camera_matrix);

	Mat isomap = Mat::zeros(isomap_resolution, isomap_resolution, CV_8UC4);
	// #Todo: We should handle gray images, but output a 4-channel isomap nevertheless I think.

	std::vector<std::future<void>> results;
	for (const auto& triangle_indices : mesh.tvi) {

		// Note: If there's a performance problem, there's no need to capture the whole mesh - we could capture only the three required vertices with their texcoords.
		auto extract_triangle = [&mesh, &affine_camera_matrix, &triangle_indices, &depthbuffer, &isomap, &mapping_type, &image, &compute_view_angle]() {

			// Find out if the current triangle is visible:
			// We do a second rendering-pass here. We use the depth-buffer of the final image, and then, here,
			// check if each pixel in a triangle is visible. If the whole triangle is visible, we use it to extract
			// the texture.
			// Possible improvement: - If only part of the triangle is visible, split it

			// This could be optimized in 2 ways though:
			// - Use render(), or as in render(...), transfer the vertices once, not in a loop over all triangles (vertices are getting transformed multiple times)
			// - We transform them later (below) a second time. Only do it once.

			cv::Vec4f v0_as_Vec4f(mesh.vertices[triangle_indices[0]].x, mesh.vertices[triangle_indices[0]].y, mesh.vertices[triangle_indices[0]].z, mesh.vertices[triangle_indices[0]].w);
			cv::Vec4f v1_as_Vec4f(mesh.vertices[triangle_indices[1]].x, mesh.vertices[triangle_indices[1]].y, mesh.vertices[triangle_indices[1]].z, mesh.vertices[triangle_indices[1]].w);
			cv::Vec4f v2_as_Vec4f(mesh.vertices[triangle_indices[2]].x, mesh.vertices[triangle_indices[2]].y, mesh.vertices[triangle_indices[2]].z, mesh.vertices[triangle_indices[2]].w);

			// Project the triangle vertices to screen coordinates, and use the depthbuffer to check whether the triangle is visible:
			const Vec4f v0 = Mat(affine_camera_matrix * Mat(v0_as_Vec4f));
			const Vec4f v1 = Mat(affine_camera_matrix * Mat(v1_as_Vec4f));
			const Vec4f v2 = Mat(affine_camera_matrix * Mat(v2_as_Vec4f));

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
				const Vec3f face_normal = calculate_face_normal(Vec3f(Mat(v0_as_Vec4f).rowRange(0, 3)), Vec3f(Mat(v1_as_Vec4f).rowRange(0, 3)), Vec3f(Mat(v2_as_Vec4f).rowRange(0, 3)));
				// Transform the normal to "screen" (kind of "eye") space using the upper 3x3 part of the affine camera matrix (=the translation can be ignored):
				Vec3f face_normal_transformed = Mat(affine_camera_matrix.rowRange(0, 3).colRange(0, 3) * Mat(face_normal));
				face_normal_transformed /= cv::norm(face_normal_transformed, cv::NORM_L2); // normalise to unit length
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

			// We now have the source triangles in the image and the source triangle in the isomap
			// We use the inverse/ backward mapping approach, so we want to find the corresponding texel (texture-pixel) for each pixel in the isomap

			// Get the inverse Affine Transform from original image: from dst (pixel in isomap) to src (in image)
			Mat warp_mat_org_inv = cv::getAffineTransform(dst_tri, src_tri);
			warp_mat_org_inv.convertTo(warp_mat_org_inv, CV_32FC1);

			// We now loop over all pixels in the triangle and select, depending on the mapping type, the corresponding texel(s) in the source image
			for (int x = min(dst_tri[0].x, min(dst_tri[1].x, dst_tri[2].x)); x < max(dst_tri[0].x, max(dst_tri[1].x, dst_tri[2].x)); ++x) {
				for (int y = min(dst_tri[0].y, min(dst_tri[1].y, dst_tri[2].y)); y < max(dst_tri[0].y, max(dst_tri[1].y, dst_tri[2].y)); ++y) {
					if (detail::is_point_in_triangle(cv::Point2f(x, y), dst_tri[0], dst_tri[1], dst_tri[2])) {

						// As the coordinates of the transformed pixel in the image will most likely not lie on a texel, we have to choose how to 
						// calculate the pixel colors depending on the next texels
						// there are three different texture interpolation methods: area, bilinear and nearest neighbour

						// Area mapping: calculate mean color of texels in transformed pixel area
						if (mapping_type == TextureInterpolation::Area) {

							// calculate positions of 4 corners of pixel in image (src)
							Vec3f homogenous_dst_upper_left(x - 0.5f, y - 0.5f, 1.0f);
							Vec3f homogenous_dst_upper_right(x + 0.5f, y - 0.5f, 1.0f);
							Vec3f homogenous_dst_lower_left(x - 0.5f, y + 0.5f, 1.0f);
							Vec3f homogenous_dst_lower_right(x + 0.5f, y + 0.5f, 1.0f);

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

							// loop over square in which quadrangle out of the four corners of pixel is
							for (int a = ceil(min_a); a <= floor(max_a); ++a)
							{
								for (int b = ceil(min_b); b <= floor(max_b); ++b)
								{
									// check if texel is in quadrangle
									if (detail::is_point_in_triangle(cv::Point2f(a, b), src_texel_upper_left, src_texel_lower_left, src_texel_upper_right) || detail::is_point_in_triangle(cv::Point2f(a, b), src_texel_lower_left, src_texel_upper_right, src_texel_lower_right)) {
										if (a < image.cols && b < image.rows) { // check if texel is in image
											num_texels++;
											color += image.at<Vec3b>(b, a);
										}
									}
								}
							}
							if (num_texels > 0)
								color = color / num_texels;
							else { // if no corresponding texel found, nearest neighbour interpolation
								// calculate corresponding position of dst_coord pixel center in image (src)
								Vec3f homogenous_dst_coord = Vec3f(x, y, 1.0f);
								Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

								if ((cvRound(src_texel[1]) < image.rows) && cvRound(src_texel[0]) < image.cols) {
									color = image.at<Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]));
								}
							}
							isomap.at<Vec3b>(y, x) = color;
						}
						// Bilinear mapping: calculate pixel color depending on the four neighbouring texels
						else if (mapping_type == TextureInterpolation::Bilinear) {

							// calculate corresponding position of dst_coord pixel center in image (src)
							Vec3f homogenous_dst_coord(x, y, 1.0f);
							Vec2f src_texel = Mat(warp_mat_org_inv * Mat(homogenous_dst_coord));

							// calculate euclidean distances to next 4 texels
							using std::sqrt;
							using std::pow;
							float distance_upper_left = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
							float distance_upper_right = sqrt(pow(src_texel[0] - floor(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));
							float distance_lower_left = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - floor(src_texel[1]), 2));
							float distance_lower_right = sqrt(pow(src_texel[0] - ceil(src_texel[0]), 2) + pow(src_texel[1] - ceil(src_texel[1]), 2));

							// normalise distances that the sum of all distances is 1
							float sum_distances = distance_lower_left + distance_lower_right + distance_upper_left + distance_upper_right;
							distance_lower_left /= sum_distances;
							distance_lower_right /= sum_distances;
							distance_upper_left /= sum_distances;
							distance_upper_right /= sum_distances;

							// set color depending on distance from next 4 texels
							for (int color = 0; color < 3; ++color) {
								float color_upper_left = image.at<Vec3b>(floor(src_texel[1]), floor(src_texel[0]))[color] * distance_upper_left;
								float color_upper_right = image.at<Vec3b>(floor(src_texel[1]), ceil(src_texel[0]))[color] * distance_upper_right;
								float color_lower_left = image.at<Vec3b>(ceil(src_texel[1]), floor(src_texel[0]))[color] * distance_lower_left;
								float color_lower_right = image.at<Vec3b>(ceil(src_texel[1]), ceil(src_texel[0]))[color] * distance_lower_right;

								isomap.at<Vec3b>(y, x)[color] = color_upper_left + color_upper_right + color_lower_left + color_lower_right;
							}
						}
						// NearestNeighbour mapping: set color of pixel to color of nearest texel
						else if (mapping_type == TextureInterpolation::NearestNeighbour) {

							// calculate corresponding position of dst_coord pixel center in image (src)
							const Mat homogenous_dst_coord(Vec3f(x, y, 1.0f));
							const Vec2f src_texel = Mat(warp_mat_org_inv * homogenous_dst_coord);

							if ((cvRound(src_texel[1]) < image.rows) && (cvRound(src_texel[0]) < image.cols) && cvRound(src_texel[0]) > 0 && cvRound(src_texel[1]) > 0)
							{
								cv::Vec4b isomap_pixel;
								isomap.at<cv::Vec4b>(y, x)[0] = image.at<Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]))[0];
								isomap.at<cv::Vec4b>(y, x)[1] = image.at<Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]))[1];
								isomap.at<cv::Vec4b>(y, x)[2] = image.at<Vec3b>(cvRound(src_texel[1]), cvRound(src_texel[0]))[2];
								isomap.at<cv::Vec4b>(y, x)[3] = static_cast<uchar>(alpha_value); // pixel is visible
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
	if (mesh.texcoords.size() <= 3448)
	{
		isomap = detail::interpolate_black_line(isomap);
	}

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
cv::Mat extract_texture(core::Mesh mesh, glm::mat4x4 view_model_matrix, glm::mat4x4 projection_matrix,
                        glm::vec4 /*viewport, not needed at the moment */, cv::Mat image,
                        bool /* compute_view_angle, unused atm */, int isomap_resolution = 512)
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
                vec4(mesh.texcoords[tvi[0]][0] * tex_width, mesh.texcoords[tvi[0]][1] * tex_height,
                     wnd_coords[tvi[0]].z /* z_ndc */, wnd_coords[tvi[0]].w /* 1/w_clip */),
                vec3(/* empty */),
                vec2(
                    wnd_coords[tvi[0]].x / image.cols,
                    /* maybe 1 - ... ? */ wnd_coords[tvi[0]].y /
                        image
                            .rows /* wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1. */)};
            detail::Vertex<double> pb{
                vec4(mesh.texcoords[tvi[1]][0] * tex_width, mesh.texcoords[tvi[1]][1] * tex_height,
                     wnd_coords[tvi[1]].z /* z_ndc */, wnd_coords[tvi[1]].w /* 1/w_clip */),
                vec3(/* empty */),
                vec2(
                    wnd_coords[tvi[1]].x / image.cols,
                    /* maybe 1 - ... ? */ wnd_coords[tvi[1]].y /
                        image
                            .rows /* wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1. */)};
            detail::Vertex<double> pc{
                vec4(mesh.texcoords[tvi[2]][0] * tex_width, mesh.texcoords[tvi[2]][1] * tex_height,
                     wnd_coords[tvi[2]].z /* z_ndc */, wnd_coords[tvi[2]].w /* 1/w_clip */),
                vec3(/* empty */),
                vec2(
                    wnd_coords[tvi[2]].x / image.cols,
                    /* maybe 1 - ... ? */ wnd_coords[tvi[2]].y /
                        image
                            .rows /* wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1. */)};
            extraction_rasterizer.raster_triangle(pa, pb, pc, image_to_extract_from_as_tex);
        }
    }

    return extraction_rasterizer.colorbuffer;
};

} /* namespace v2 */

namespace detail {

// Workaround for the pixels that don't get filled in extract_texture().
// There's a vertical line of missing values in the middle of the isomap,
// as well as a few pixels on a horizontal line around the mouth. They
// manifest themselves as black lines in the final isomap. This function
// just fills these missing values by interpolating between two neighbouring
// pixels. See GitHub issue #4.
inline cv::Mat interpolate_black_line(cv::Mat isomap)
{
	assert(isomap.type() == CV_8UC4);
	// Replace the vertical black line ("missing data"):
	int col = isomap.cols / 2;
	for (int row = 0; row < isomap.rows; ++row)
	{
		if (isomap.at<cv::Vec4b>(row, col) == cv::Vec4b(0, 0, 0, 0))
		{
			isomap.at<cv::Vec4b>(row, col) = isomap.at<cv::Vec4b>(row, col - 1) * 0.5f + isomap.at<cv::Vec4b>(row, col + 1) * 0.5f;
		}
	}

	// Replace the horizontal line around the mouth that occurs in the
	// isomaps of resolution 512x512 and higher:
	if (isomap.rows == 512) // num cols is 512 as well
	{
		int r = 362;
		for (int c = 206; c <= 306; ++c)
		{
			if (isomap.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
			{
				isomap.at<cv::Vec4b>(r, c) = isomap.at<cv::Vec4b>(r - 1, c) * 0.5f + isomap.at<cv::Vec4b>(r + 1, c) * 0.5f;
			}
		}
	}
	if (isomap.rows == 1024) // num cols is 1024 as well
	{
		int r = 724;
		for (int c = 437; c <= 587; ++c)
		{
			if (isomap.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
			{
				isomap.at<cv::Vec4b>(r, c) = isomap.at<cv::Vec4b>(r - 1, c) * 0.5f + isomap.at<cv::Vec4b>(r + 1, c) * 0.5f;
			}
		}
		r = 725;
		for (int c = 411; c <= 613; ++c)
		{
			if (isomap.at<cv::Vec4b>(r, c) == cv::Vec4b(0, 0, 0, 0))
			{
				isomap.at<cv::Vec4b>(r, c) = isomap.at<cv::Vec4b>(r - 1, c) * 0.5f + isomap.at<cv::Vec4b>(r + 1, c) * 0.5f;
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
