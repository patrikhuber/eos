/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/draw_utils.hpp
 *
 * Copyright 2017 Patrik Huber
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

#ifndef RENDER_DRAW_UTILS_HPP_
#define RENDER_DRAW_UTILS_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/render_detail.hpp"

#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace eos {
	namespace render {

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] color Colour of the mesh to be drawn.
 */
inline void draw_wireframe(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
		const auto p2 = glm::project({ mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2] }, modelview, projection, viewport);
		const auto p3 = glm::project({ mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2] }, modelview, projection, viewport);
		if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
		{
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), color);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), color);
		}
	}
};

/**
 * Draws the texture coordinates (uv-coords) of the given mesh
 * into an image by looping over the triangles and drawing each
 * triangle's texcoords.
 *
 * Note/Todo: This function has a slight problems, the lines do not actually get
 * drawn blue, if the image is 8UC4. Well if I save a PNG, it is blue. Not sure.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] image An optional image to draw onto.
 * @return An image with the texture coordinate triangles drawn in it, 512x512 if no image is given.
 */
inline cv::Mat draw_texcoords(core::Mesh mesh, cv::Mat image = cv::Mat())
{
	using cv::Point2f;
	using cv::Scalar;
	if (image.empty())
	{
		image = cv::Mat(512, 512, CV_8UC4, Scalar(0.0f, 0.0f, 0.0f, 255.0f));
	}

	for (const auto& triIdx : mesh.tvi) {
		cv::line(image, Point2f(mesh.texcoords[triIdx[0]][0] * image.cols, mesh.texcoords[triIdx[0]][1] * image.rows), Point2f(mesh.texcoords[triIdx[1]][0] * image.cols, mesh.texcoords[triIdx[1]][1] * image.rows), Scalar(255.0f, 0.0f, 0.0f));
		cv::line(image, Point2f(mesh.texcoords[triIdx[1]][0] * image.cols, mesh.texcoords[triIdx[1]][1] * image.rows), Point2f(mesh.texcoords[triIdx[2]][0] * image.cols, mesh.texcoords[triIdx[2]][1] * image.rows), Scalar(255.0f, 0.0f, 0.0f));
		cv::line(image, Point2f(mesh.texcoords[triIdx[2]][0] * image.cols, mesh.texcoords[triIdx[2]][1] * image.rows), Point2f(mesh.texcoords[triIdx[0]][0] * image.cols, mesh.texcoords[triIdx[0]][1] * image.rows), Scalar(255.0f, 0.0f, 0.0f));
	}
	return image;
};

	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_DRAW_UTILS_HPP_ */
