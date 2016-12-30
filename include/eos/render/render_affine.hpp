/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/render_affine.hpp
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

#ifndef RENDER_AFFINE_HPP_
#define RENDER_AFFINE_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/render/detail/render_affine_detail.hpp"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "opencv2/core/core.hpp"

#include <utility>

namespace eos {
	namespace render {

/**
 * Renders the mesh using the given affine camera matrix and returns the colour and depth buffer images.
 * The camera matrix should be one estimated with fitting::estimate_affine_camera (Hartley & Zisserman algorithm).
 *
 * If the given mesh is a shape-only mesh without vertex-colour information, the vertices will be rendered in grey.
 *
 * #Todo: May consider an overload where we pass in an image, use that as colourbuffer and draw over it.
 * #Todo: Add texture rendering to this. Then, create an additional function in extract_texure that is fully optimised for only the extraction.
 *
 * @param[in] mesh A 3D mesh.
 * @param[in] affine_camera_matrix 3x4 affine camera matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] do_backface_culling Whether the renderer should perform backface culling.
 * @return A pair with the colourbuffer as its first element and the depthbuffer as the second element.
 */
inline std::pair<cv::Mat, cv::Mat> render_affine(core::Mesh mesh, cv::Mat affine_camera_matrix, int viewport_width, int viewport_height, bool do_backface_culling = true)
{
	assert(mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or, alternatively, it has to be a shape-only model.
	//assert(mesh.vertices.size() == mesh.texcoords.size() || mesh.texcoords.empty()); // same for the texcoords

	using cv::Mat;
	using std::vector;

	Mat colourbuffer = Mat::zeros(viewport_height, viewport_width, CV_8UC4);
	Mat depthbuffer = std::numeric_limits<float>::max() * Mat::ones(viewport_height, viewport_width, CV_64FC1);

	Mat affine_with_z = detail::calculate_affine_z_direction(affine_camera_matrix);

	vector<detail::Vertex> projected_vertices;
	projected_vertices.reserve(mesh.vertices.size());
	for (int i = 0; i < mesh.vertices.size(); ++i) {
		Mat vertex_screen_coords = affine_with_z * Mat(cv::Vec4f(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z, mesh.vertices[i].w));
		glm::tvec4<float> vertex_screen_coords_glm(vertex_screen_coords.at<float>(0), vertex_screen_coords.at<float>(1), vertex_screen_coords.at<float>(2), vertex_screen_coords.at<float>(3));
		glm::tvec3<float> vertex_colour;
		if (mesh.colors.empty()) {
			vertex_colour = glm::tvec3<float>(0.5f, 0.5f, 0.5f);
		}
		else {
			vertex_colour = mesh.colors[i];
		}
		projected_vertices.push_back(detail::Vertex(vertex_screen_coords_glm, vertex_colour, mesh.texcoords[i]));
	}

	// All vertices are screen-coordinates now
	vector<detail::TriangleToRasterize> triangles_to_raster;
	for (const auto& tri_indices : mesh.tvi) {
		if (do_backface_culling) {
			if (!detail::are_vertices_ccw_in_screen_space(glm::tvec2<float>(projected_vertices[tri_indices[0]].position), glm::tvec2<float>(projected_vertices[tri_indices[1]].position), glm::tvec2<float>(projected_vertices[tri_indices[2]].position)))
				continue; // don't render this triangle
		}

		// Get the bounding box of the triangle:
		// take care: What do we do if all 3 vertices are not visible. Seems to work on a test case.
		cv::Rect bounding_box = detail::calculate_clipped_bounding_box(glm::tvec2<float>(projected_vertices[tri_indices[0]].position), glm::tvec2<float>(projected_vertices[tri_indices[1]].position), glm::tvec2<float>(projected_vertices[tri_indices[2]].position), viewport_width, viewport_height);
		auto min_x = bounding_box.x;
		auto max_x = bounding_box.x + bounding_box.width;
		auto min_y = bounding_box.y;
		auto max_y = bounding_box.y + bounding_box.height;

		if (max_x <= min_x || max_y <= min_y) // Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
			continue;

		detail::TriangleToRasterize t;
		t.min_x = min_x;
		t.max_x = max_x;
		t.min_y = min_y;
		t.max_y = max_y;
		t.v0 = projected_vertices[tri_indices[0]];
		t.v1 = projected_vertices[tri_indices[1]];
		t.v2 = projected_vertices[tri_indices[2]];

		triangles_to_raster.push_back(t);
	}

	// Raster all triangles, i.e. colour the pixel values and write the z-buffer
	for (auto&& triangle : triangles_to_raster) {
		detail::raster_triangle_affine(triangle, colourbuffer, depthbuffer);
	}
	return std::make_pair(colourbuffer, depthbuffer);
};


	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_AFFINE_HPP_ */
