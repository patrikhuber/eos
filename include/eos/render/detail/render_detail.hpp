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
#include "eos/render/utils.hpp"
#include "eos/render/Texture.hpp"
#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/TriangleToRasterize.hpp"

#include "glm/glm.hpp" // tvec2, glm::precision, tvec3, tvec4, normalize, dot, cross

#include "Eigen/Core"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
	namespace render {
		namespace detail {

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
template<typename T, glm::precision P = glm::defaultp>
Rect<int> calculate_clipped_bounding_box(const glm::tvec2<T, P>& v0, const glm::tvec2<T, P>& v1, const glm::tvec2<T, P>& v2, int viewport_width, int viewport_height)
{
	/* Old, producing artifacts:
	t.minX = max(min(t.v0.position[0], min(t.v1.position[0], t.v2.position[0])), 0.0f);
	t.maxX = min(max(t.v0.position[0], max(t.v1.position[0], t.v2.position[0])), (float)(viewportWidth - 1));
	t.minY = max(min(t.v0.position[1], min(t.v1.position[1], t.v2.position[1])), 0.0f);
	t.maxY = min(max(t.v0.position[1], max(t.v1.position[1], t.v2.position[1])), (float)(viewportHeight - 1));*/

	using std::min;
	using std::max;
	using std::floor;
	using std::ceil;
	int minX = max(min(floor(v0[0]), min(floor(v1[0]), floor(v2[0]))), T(0)); // Readded this comment after merge: What about rounding, or rather the conversion from double to int?
	int maxX = min(max(ceil(v0[0]), max(ceil(v1[0]), ceil(v2[0]))), static_cast<T>(viewport_width - 1));
	int minY = max(min(floor(v0[1]), min(floor(v1[1]), floor(v2[1]))), T(0));
	int maxY = min(max(ceil(v0[1]), max(ceil(v1[1]), ceil(v2[1]))), static_cast<T>(viewport_height - 1));
	return Rect<int>{ minX, minY, maxX - minX, maxY - minY };
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
template<typename T, glm::precision P = glm::defaultp>
bool are_vertices_ccw_in_screen_space(const glm::tvec2<T, P>& v0, const glm::tvec2<T, P>& v1, const glm::tvec2<T, P>& v2)
{
	const auto dx01 = v1[0] - v0[0]; // todo: replace with x/y (GLM)
	const auto dy01 = v1[1] - v0[1];
	const auto dx02 = v2[0] - v0[0];
	const auto dy02 = v2[1] - v0[1];

	return (dx01*dy02 - dy01*dx02 < T(0)); // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
};

template<typename T, glm::precision P = glm::defaultp>
double implicit_line(float x, float y, const glm::tvec4<T, P>& v1, const glm::tvec4<T, P>& v2)
{
	return ((double)v1[1] - (double)v2[1])*(double)x + ((double)v2[0] - (double)v1[0])*(double)y + (double)v1[0] * (double)v2[1] - (double)v2[0] * (double)v1[1];
};

inline std::vector<Vertex<float>> clip_polygon_to_plane_in_4d(const std::vector<Vertex<float>>& vertices, const glm::tvec4<float>& plane_normal)
{
	std::vector<Vertex<float>> clippedVertices;

	// We can have 2 cases:
	//	* 1 vertex visible: we make 1 new triangle out of the visible vertex plus the 2 intersection points with the near-plane
	//  * 2 vertices visible: we have a quad, so we have to make 2 new triangles out of it.

	// See here for more info? http://math.stackexchange.com/questions/400268/equation-for-a-line-through-a-plane-in-homogeneous-coordinates

	for (unsigned int i = 0; i < vertices.size(); i++)
	{
		int a = i; // the current vertex
		int b = (i + 1) % vertices.size(); // the following vertex (wraps around 0)

		float fa = glm::dot(vertices[a].position, plane_normal); // Note: Shouldn't they be unit length?
		float fb = glm::dot(vertices[b].position, plane_normal); // < 0 means on visible side, > 0 means on invisible side?

		if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) // one vertex is on the visible side of the plane, one on the invisible? so we need to split?
		{
			auto direction = vertices[b].position - vertices[a].position;
			float t = -(glm::dot(plane_normal, vertices[a].position)) / (glm::dot(plane_normal, direction)); // the parametric value on the line, where the line to draw intersects the plane?

			// generate a new vertex at the line-plane intersection point
			auto position = vertices[a].position + t*direction;
			auto color = vertices[a].color + t*(vertices[b].color - vertices[a].color);
			auto texCoord = vertices[a].texcoords + t*(vertices[b].texcoords - vertices[a].texcoords);	// We could omit that if we don't render with texture.

			if (fa < 0) // we keep the original vertex plus the new one
			{
				clippedVertices.push_back(vertices[a]);
				clippedVertices.push_back(Vertex<float>{position, color, texCoord});
			}
			else if (fb < 0) // we use only the new vertex
			{
				clippedVertices.push_back(Vertex<float>{position, color, texCoord});
			}
		}
		else if (fa < 0 && fb < 0) // both are visible (on the "good" side of the plane), no splitting required, use the current vertex
		{
			clippedVertices.push_back(vertices[a]);
		}
		// else, both vertices are not visible, nothing to add and draw
	}

	return clippedVertices;
};

/**
 * @brief Todo.
 *
 * Takes in clip coords? and outputs NDC.
 * divides by w and outputs [x_ndc, y_ndc, z_ndc, 1/w_clip].
 * The w-component is set to 1/w_clip (which is what OpenGL passes to the FragmentShader).
 *
 * @param[in] vertex X.
 * @ return X.
 */
template <typename T, glm::precision P = glm::defaultp>
glm::tvec4<T, P> divide_by_w(const glm::tvec4<T, P>& vertex)
{
    auto one_over_w = 1.0 / vertex.w;
    // divide by w: (if ortho, w will just be 1)
    glm::tvec4<T, P> v_ndc(vertex / vertex.w);
    // Set the w coord to 1/w (i.e. 1/w_clip). This is what OpenGL passes to the FragmentShader.
    v_ndc.w = one_over_w;
    return v_ndc;
};

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
