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

// used only in tex2D_linear_mipmap_linear
// template?
float clamp(float x, float a, float b)
{
	return std::max(std::min(x, b), a);
};

cv::Vec2f texcoord_wrap(const cv::Vec2f& texcoords)
{
	return cv::Vec2f(texcoords[0] - (int)texcoords[0], texcoords[1] - (int)texcoords[1]);
};

// forward decls
cv::Vec3f tex2d_linear_mipmap_linear(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy);
cv::Vec3f tex2d_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex, const Texture& texture);

cv::Vec3f tex2d(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy)
{
	return (1.0f / 255.0f) * tex2d_linear_mipmap_linear(texcoords, texture, dudx, dudy, dvdx, dvdy);
};

cv::Vec3f tex2d_linear_mipmap_linear(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy)
{
	using cv::Vec2f;
	float px = std::sqrt(std::pow(dudx, 2) + std::pow(dvdx, 2));
	float py = std::sqrt(std::pow(dudy, 2) + std::pow(dvdy, 2));
	float lambda = std::log(std::max(px, py)) / CV_LOG2;
	unsigned char mipmapIndex1 = detail::clamp((int)lambda, 0.0f, std::max(texture.widthLog, texture.heightLog) - 1);
	unsigned char mipmapIndex2 = mipmapIndex1 + 1;

	Vec2f imageTexCoord = detail::texcoord_wrap(texcoords);
	Vec2f imageTexCoord1 = imageTexCoord;
	imageTexCoord1[0] *= texture.mipmaps[mipmapIndex1].cols;
	imageTexCoord1[1] *= texture.mipmaps[mipmapIndex1].rows;
	Vec2f imageTexCoord2 = imageTexCoord;
	imageTexCoord2[0] *= texture.mipmaps[mipmapIndex2].cols;
	imageTexCoord2[1] *= texture.mipmaps[mipmapIndex2].rows;

	cv::Vec3f color, color1, color2;
	color1 = tex2d_linear(imageTexCoord1, mipmapIndex1, texture);
	color2 = tex2d_linear(imageTexCoord2, mipmapIndex2, texture);
	float lambdaFrac = std::max(lambda, 0.0f);
	lambdaFrac = lambdaFrac - (int)lambdaFrac;
	color = (1.0f - lambdaFrac)*color1 + lambdaFrac*color2;

	return color;
};

cv::Vec3f tex2d_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmap_index, const Texture& texture)
{
	int x = (int)imageTexCoord[0];
	int y = (int)imageTexCoord[1];
	float alpha = imageTexCoord[0] - x;
	float beta = imageTexCoord[1] - y;
	float oneMinusAlpha = 1.0f - alpha;
	float oneMinusBeta = 1.0f - beta;
	float a = oneMinusAlpha * oneMinusBeta;
	float b = alpha * oneMinusBeta;
	float c = oneMinusAlpha * beta;
	float d = alpha * beta;
	cv::Vec3f color;

	using cv::Vec4b;
	//int pixelIndex;
	//pixelIndex = getPixelIndex_wrap(x, y, texture->mipmaps[mipmapIndex].cols, texture->mipmaps[mipmapIndex].rows);
	int pixelIndexCol = x; if (pixelIndexCol == texture.mipmaps[mipmap_index].cols) { pixelIndexCol = 0; }
	int pixelIndexRow = y; if (pixelIndexRow == texture.mipmaps[mipmap_index].rows) { pixelIndexRow = 0; }
	//std::cout << texture.mipmaps[mipmapIndex].cols << " " << texture.mipmaps[mipmapIndex].rows << " " << texture.mipmaps[mipmapIndex].channels() << std::endl;
	//cv::imwrite("mm.png", texture.mipmaps[mipmapIndex]);
	color[0] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y, texture.mipmaps[mipmapIndex].cols, texture.mipmaps[mipmapIndex].rows);
	pixelIndexCol = x + 1; if (pixelIndexCol == texture.mipmaps[mipmap_index].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y; if (pixelIndexRow == texture.mipmaps[mipmap_index].rows) { pixelIndexRow = 0; }
	color[0] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x, y + 1, texture.mipmaps[mipmapIndex].cols, texture.mipmaps[mipmapIndex].rows);
	pixelIndexCol = x; if (pixelIndexCol == texture.mipmaps[mipmap_index].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y + 1; if (pixelIndexRow == texture.mipmaps[mipmap_index].rows) { pixelIndexRow = 0; }
	color[0] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	//pixelIndex = getPixelIndex_wrap(x + 1, y + 1, texture.mipmaps[mipmapIndex].cols, texture.mipmaps[mipmapIndex].rows);
	pixelIndexCol = x + 1; if (pixelIndexCol == texture.mipmaps[mipmap_index].cols) { pixelIndexCol = 0; }
	pixelIndexRow = y + 1; if (pixelIndexRow == texture.mipmaps[mipmap_index].rows) { pixelIndexRow = 0; }
	color[0] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
	color[1] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
	color[2] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

	return color;
};

void raster_triangle(TriangleToRasterize triangle, cv::Mat colourbuffer, cv::Mat depthbuffer, boost::optional<Texture> texture, bool enable_far_clipping)
{
	using cv::Vec2f;
	using cv::Vec3f;
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
				// The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane earlier.
				// TODO: Use enable_far_clipping here.
				bool draw = true;
				if (enable_far_clipping)
				{
					if (z_affine > 1.0)
					{
						draw = false;
					}
				}
				//if (z_affine < depthbuffer.at<double>(pixelIndexRow, pixelIndexCol)/* && z_affine <= 1.0*/) // what to do in ortho case without n/f "squashing"? should we always squash? or a flag?
				if (z_affine < depthbuffer.at<double>(pixel_index_row, pixel_index_col) && draw)
				{
					// perspective-correct barycentric weights
					double d = alpha*triangle.one_over_z0 + beta*triangle.one_over_z1 + gamma*triangle.one_over_z2;
					d = 1.0 / d;
					alpha *= d*triangle.one_over_z0; // In case of affine cam matrix, everything is 1 and a/b/g don't get changed.
					beta *= d*triangle.one_over_z1;
					gamma *= d*triangle.one_over_z2;

					// attributes interpolation
					Vec3f color_persp = alpha*triangle.v0.color + beta*triangle.v1.color + gamma*triangle.v2.color;
					Vec2f texcoords_persp = alpha*triangle.v0.texcoords + beta*triangle.v1.texcoords + gamma*triangle.v2.texcoords;

					Vec3f pixel_color;
					// Pixel Shader:
					if (texture) {	// We use texturing
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
						Vec3f texture_color = detail::tex2d(texcoords_persp, texture.get(), dudx, dudy, dvdx, dvdy); // uses the current texture
						pixel_color = Vec3f(texture_color[2], texture_color[1], texture_color[0]);
						// other: color.mul(tex2D(texture, texCoord));
						// Old note: for texturing, we load the texture as BGRA, so the colors get the wrong way in the next few lines...
					}
					else {	// We use vertex-coloring
						// color_persp is in RGB
						pixel_color = color_persp;
					}

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
