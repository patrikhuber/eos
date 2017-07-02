/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/render_detail.hpp
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

#ifndef RENDER_DETAIL_HPP_
#define RENDER_DETAIL_HPP_

#include "eos/core/Image.hpp"
#include "eos/render/Rect.hpp"
#include "eos/render/Texture.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/TriangleToRasterize.hpp"
#include "eos/render/detail/render_detail_utils.hpp"
#include "eos/render/detail/texturing.hpp"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

#include "boost/optional.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
	namespace render {
		namespace detail {

// Todo: Split this function into the general (core-part) and the texturing part.
// Then, utils::extractTexture can re-use the core-part.
// Note: Maybe a bit outdated "todo" above.
inline boost::optional<TriangleToRasterize> process_prospective_tri(Vertex<float> v0, Vertex<float> v1, Vertex<float> v2, int viewport_width, int viewport_height, bool enable_backface_culling)
{
	using glm::vec2;
	using glm::vec3;

	TriangleToRasterize t;
	t.v0 = v0;	// no memcopy I think. the transformed vertices don't get copied and exist only once. They are a local variable in runVertexProcessor(), the ref is passed here, and if we need to rasterize it, it gets push_back'ed (=copied?) to trianglesToRasterize. Perfect I think. TODO: Not anymore, no ref here
	t.v1 = v1;
	t.v2 = v2;

	// Only for texturing or perspective texturing:
	//t.texture = _texture;
	t.one_over_z0 = 1.0 / (double)t.v0.position[3];
	t.one_over_z1 = 1.0 / (double)t.v1.position[3];
	t.one_over_z2 = 1.0 / (double)t.v2.position[3];

	// divide by w
	// if ortho, we can do the divide as well, it will just be a / 1.0f.
	t.v0.position = t.v0.position / t.v0.position[3];
	t.v1.position = t.v1.position / t.v1.position[3];
	t.v2.position = t.v2.position / t.v2.position[3];

	// project from 4D to 2D window position with depth value in z coordinate
	// Viewport transform:
	/* (a possible optimisation might be to use matrix multiplication for this as well
	   and do it for all triangles at once? See 'windowTransform' in:
	   https://github.com/elador/FeatureDetection/blob/964f0b2107ce73ef2f06dc829e5084be421de5a5/libRender/src/render/RenderDevice.cpp)
	*/
	vec2 v0_screen = clip_to_screen_space(vec2(t.v0.position[0], t.v0.position[1]), viewport_width, viewport_height);
	t.v0.position[0] = v0_screen[0];
	t.v0.position[1] = v0_screen[1];
	vec2 v1_screen = clip_to_screen_space(vec2(t.v1.position[0], t.v1.position[1]), viewport_width, viewport_height);
	t.v1.position[0] = v1_screen[0];
	t.v1.position[1] = v1_screen[1];
	vec2 v2_screen = clip_to_screen_space(vec2(t.v2.position[0], t.v2.position[1]), viewport_width, viewport_height);
	t.v2.position[0] = v2_screen[0];
	t.v2.position[1] = v2_screen[1];

	if (enable_backface_culling) {
		if (!are_vertices_ccw_in_screen_space(glm::tvec2<float>(t.v0.position), glm::tvec2<float>(t.v1.position), glm::tvec2<float>(t.v2.position)))
			return boost::none;
	}

	// Get the bounding box of the triangle:
	Rect<int> boundingBox = calculate_clipped_bounding_box(glm::tvec2<float>(t.v0.position), glm::tvec2<float>(t.v1.position), glm::tvec2<float>(t.v2.position), viewport_width, viewport_height);
	t.min_x = boundingBox.x;
	t.max_x = boundingBox.x + boundingBox.width;
	t.min_y = boundingBox.y;
	t.max_y = boundingBox.y + boundingBox.height;

	if (t.max_x <= t.min_x || t.max_y <= t.min_y) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		return boost::none;

	// Which of these is for texturing, mipmapping, what for perspective?
	// for partial derivatives computation
	t.alphaPlane = plane(vec3(t.v0.position[0], t.v0.position[1], t.v0.texcoords[0] * t.one_over_z0),
		vec3(t.v1.position[0], t.v1.position[1], t.v1.texcoords[0] * t.one_over_z1),
		vec3(t.v2.position[0], t.v2.position[1], t.v2.texcoords[0] * t.one_over_z2));
	t.betaPlane = plane(vec3(t.v0.position[0], t.v0.position[1], t.v0.texcoords[1] * t.one_over_z0),
		vec3(t.v1.position[0], t.v1.position[1], t.v1.texcoords[1] * t.one_over_z1),
		vec3(t.v2.position[0], t.v2.position[1], t.v2.texcoords[1] * t.one_over_z2));
	t.gammaPlane = plane(vec3(t.v0.position[0], t.v0.position[1], t.one_over_z0),
		vec3(t.v1.position[0], t.v1.position[1], t.one_over_z1),
		vec3(t.v2.position[0], t.v2.position[1], t.one_over_z2));
	t.one_over_alpha_c = 1.0f / t.alphaPlane.c;
	t.one_over_beta_c = 1.0f / t.betaPlane.c;
	t.one_over_gamma_c = 1.0f / t.gammaPlane.c;
	t.alpha_ffx = -t.alphaPlane.a * t.one_over_alpha_c;
	t.beta_ffx = -t.betaPlane.a * t.one_over_beta_c;
	t.gamma_ffx = -t.gammaPlane.a * t.one_over_gamma_c;
	t.alpha_ffy = -t.alphaPlane.b * t.one_over_alpha_c;
	t.beta_ffy = -t.betaPlane.b * t.one_over_beta_c;
	t.gamma_ffy = -t.gammaPlane.b * t.one_over_gamma_c;

	// Use t
	return boost::optional<TriangleToRasterize>(t);
};

inline void raster_triangle(TriangleToRasterize triangle, core::Image4u& colorbuffer, core::Image1d& depthbuffer, boost::optional<Texture> texture, bool enable_far_clipping)
{
	for (int yi = triangle.min_y; yi <= triangle.max_y; ++yi)
	{
		for (int xi = triangle.min_x; xi <= triangle.max_x; ++xi)
		{
			// we want centers of pixels to be used in computations. Todo: Do we?
			const float x = static_cast<float>(xi) + 0.5f;
			const float y = static_cast<float>(yi) + 0.5f;

			// these will be used for barycentric weights computation
			const double one_over_v0ToLine12 = 1.0 / implicit_line(triangle.v0.position[0], triangle.v0.position[1], triangle.v1.position, triangle.v2.position);
			const double one_over_v1ToLine20 = 1.0 / implicit_line(triangle.v1.position[0], triangle.v1.position[1], triangle.v2.position, triangle.v0.position);
			const double one_over_v2ToLine01 = 1.0 / implicit_line(triangle.v2.position[0], triangle.v2.position[1], triangle.v0.position, triangle.v1.position);
			// affine barycentric weights
			double alpha = implicit_line(x, y, triangle.v1.position, triangle.v2.position) * one_over_v0ToLine12;
			double beta = implicit_line(x, y, triangle.v2.position, triangle.v0.position) * one_over_v1ToLine20;
			double gamma = implicit_line(x, y, triangle.v0.position, triangle.v1.position) * one_over_v2ToLine01;

			// if pixel (x, y) is inside the triangle or on one of its edges
			if (alpha >= 0 && beta >= 0 && gamma >= 0)
			{
				const int pixel_index_row = yi;
				const int pixel_index_col = xi;

				const double z_affine = alpha*static_cast<double>(triangle.v0.position[2]) + beta*static_cast<double>(triangle.v1.position[2]) + gamma*static_cast<double>(triangle.v2.position[2]);
				
				bool draw = true;
				if (enable_far_clipping)
				{
					if (z_affine > 1.0)
					{
						draw = false;
					}
				}
				// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
				//if (z_affine < depthbuffer.at<double>(pixelIndexRow, pixelIndexCol)/* && z_affine <= 1.0*/) // what to do in ortho case without n/f "squashing"? should we always squash? or a flag?
				if (z_affine < depthbuffer(pixel_index_row, pixel_index_col) && draw)
				{
					// perspective-correct barycentric weights
					double d = alpha*triangle.one_over_z0 + beta*triangle.one_over_z1 + gamma*triangle.one_over_z2;
					d = 1.0 / d;
					alpha *= d*triangle.one_over_z0; // In case of affine cam matrix, everything is 1 and a/b/g don't get changed.
					beta *= d*triangle.one_over_z1;
					gamma *= d*triangle.one_over_z2;

					// attributes interpolation
					glm::tvec3<float> color_persp = static_cast<float>(alpha)*triangle.v0.color + static_cast<float>(beta)*triangle.v1.color + static_cast<float>(gamma)*triangle.v2.color; // Note: color might be empty if we use texturing and the shape-only model - but it works nonetheless? I think I set the vertex-colour to 127 in the shape-only model.
					glm::tvec2<float> texcoords_persp = static_cast<float>(alpha)*triangle.v0.texcoords + static_cast<float>(beta)*triangle.v1.texcoords + static_cast<float>(gamma)*triangle.v2.texcoords;

					glm::tvec3<float> pixel_color;
					// Pixel Shader:
					if (texture) { // We use texturing
						// check if texture != NULL?
						// partial derivatives (for mip-mapping)
						const float u_over_z = -(triangle.alphaPlane.a*x + triangle.alphaPlane.b*y + triangle.alphaPlane.d) * triangle.one_over_alpha_c;
						const float v_over_z = -(triangle.betaPlane.a*x + triangle.betaPlane.b*y + triangle.betaPlane.d) * triangle.one_over_beta_c;
						const float one_over_z = -(triangle.gammaPlane.a*x + triangle.gammaPlane.b*y + triangle.gammaPlane.d) * triangle.one_over_gamma_c;
						const float one_over_squared_one_over_z = 1.0f / std::pow(one_over_z, 2);

						// partial derivatives of U/V coordinates with respect to X/Y pixel's screen coordinates
						float dudx = one_over_squared_one_over_z * (triangle.alpha_ffx * one_over_z - u_over_z * triangle.gamma_ffx);
						float dudy = one_over_squared_one_over_z * (triangle.beta_ffx * one_over_z - v_over_z * triangle.gamma_ffx);
						float dvdx = one_over_squared_one_over_z * (triangle.alpha_ffy * one_over_z - u_over_z * triangle.gamma_ffy);
						float dvdy = one_over_squared_one_over_z * (triangle.beta_ffy * one_over_z - v_over_z * triangle.gamma_ffy);

						dudx *= texture.get().mipmaps[0].cols;
						dudy *= texture.get().mipmaps[0].cols;
						dvdx *= texture.get().mipmaps[0].rows;
						dvdy *= texture.get().mipmaps[0].rows;

						// The Texture is in BGR, thus tex2D returns BGR
						glm::tvec3<float> texture_color = detail::tex2d(texcoords_persp, texture.get(), dudx, dudy, dvdx, dvdy); // uses the current texture
						pixel_color = glm::tvec3<float>(texture_color[2], texture_color[1], texture_color[0]);
						// other: color.mul(tex2D(texture, texCoord));
						// Old note: for texturing, we load the texture as BGRA, so the colors get the wrong way in the next few lines...
					}
					else { // We use vertex-coloring
						// color_persp is in RGB
						pixel_color = color_persp;
					}

					// clamp bytes to 255
					const unsigned char red = static_cast<unsigned char>(255.0f * std::min(pixel_color[0], 1.0f)); // Todo: Proper casting (rounding?)
					const unsigned char green = static_cast<unsigned char>(255.0f * std::min(pixel_color[1], 1.0f));
					const unsigned char blue = static_cast<unsigned char>(255.0f * std::min(pixel_color[2], 1.0f));

					// update buffers
					colorbuffer(pixel_index_row, pixel_index_col)[0] = blue;
					colorbuffer(pixel_index_row, pixel_index_col)[1] = green;
					colorbuffer(pixel_index_row, pixel_index_col)[2] = red;
					colorbuffer(pixel_index_row, pixel_index_col)[3] = 255; // alpha channel
					depthbuffer(pixel_index_row, pixel_index_col) = z_affine;
				}
			}
		}
	}
};

		} /* namespace detail */
	} /* namespace render */
} /* namespace eos */

#endif /* RENDER_DETAIL_HPP_ */
