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

#include "eos/render/utils.hpp"
#include "eos/render/Texture.hpp"
#include "eos/render/detail/Vertex.hpp"

#include "glm/glm.hpp" // tvec2, glm::precision, tvec3, tvec4, normalize, dot, cross

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
	namespace render {
		namespace detail {

class plane
{
public:
	plane() {}

	plane(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
	}

	plane(const cv::Vec3f& normal, float d = 0.0f)
	{
		this->a = normal[0];
		this->b = normal[1];
		this->c = normal[2];
		this->d = d;
	}

	plane(const cv::Vec3f& point, const cv::Vec3f& normal)
	{
		a = normal[0];
		b = normal[1];
		c = normal[2];
		d = -(point.dot(normal));
	}

	plane(const cv::Vec3f& point1, const cv::Vec3f& point2, const cv::Vec3f& point3)
	{
		cv::Vec3f v1 = point2 - point1;
		cv::Vec3f v2 = point3 - point1;
		cv::Vec3f normal = (v1.cross(v2));
		normal /= cv::norm(normal, cv::NORM_L2);

		a = normal[0];
		b = normal[1];
		c = normal[2];
		d = -(point1.dot(normal));
	}

	template<typename T, glm::precision P = glm::defaultp>
	plane(const glm::tvec3<T, P>& point1, const glm::tvec3<T, P>& point2, const glm::tvec3<T, P>& point3)
	{
		glm::tvec3<T, P> v1 = point2 - point1;
		glm::tvec3<T, P> v2 = point3 - point1;
		glm::tvec3<T, P> normal = glm::cross(v1, v2);
		normal = glm::normalize(normal);

		a = normal[0];
		b = normal[1];
		c = normal[2];
		d = -glm::dot(point1, normal);
	}

	void normalize()
	{
		float length = sqrt(a*a + b*b + c*c);

		a /= length;
		b /= length;
		c /= length;
	}

	float getSignedDistanceFromPoint(const cv::Vec3f& point) const
	{
		return a*point[0] + b*point[1] + c*point[2] + d;
	}

	float getSignedDistanceFromPoint(const cv::Vec4f& point) const
	{
		return a*point[0] + b*point[1] + c*point[2] + d;
	}

public:
	float a, b, c;
	float d;
};

/**
 * A representation for a triangle that is to be rasterised.
 * Stores the enclosing bounding box of the triangle that is
 * calculated during rendering and used during rasterisation.
 *
 * Used in render_affine and render.
 */
struct TriangleToRasterize
{
	Vertex<float> v0, v1, v2;
	int min_x;
	int max_x;
	int min_y;
	int max_y;
	// Everything below is only used in the "normal" renderer, but not
	// in render_affine.
	double one_over_z0;
	double one_over_z1;
	double one_over_z2;
	//double one_over_v0ToLine12;
	//double one_over_v1ToLine20;
	//double one_over_v2ToLine01;
	plane alphaPlane;
	plane betaPlane;
	plane gammaPlane;
	double one_over_alpha_c; // those are only used for texturing -> float
	double one_over_beta_c;
	double one_over_gamma_c;
	float alpha_ffx;
	float beta_ffx;
	float gamma_ffx;
	float alpha_ffy;
	float beta_ffy;
	float gamma_ffy;
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
template<typename T, glm::precision P = glm::defaultp>
cv::Rect calculate_clipped_bounding_box(const glm::tvec2<T, P>& v0, const glm::tvec2<T, P>& v1, const glm::tvec2<T, P>& v2, int viewport_width, int viewport_height)
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

// used only in tex2D_linear_mipmap_linear
// template?
inline float clamp(float x, float a, float b)
{
	return std::max(std::min(x, b), a);
};

inline cv::Vec2f texcoord_wrap(const cv::Vec2f& texcoords)
{
	return cv::Vec2f(texcoords[0] - (int)texcoords[0], texcoords[1] - (int)texcoords[1]);
};

// forward decls
cv::Vec3f tex2d_linear_mipmap_linear(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy);
cv::Vec3f tex2d_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmapIndex, const Texture& texture);

inline cv::Vec3f tex2d(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy)
{
	return (1.0f / 255.0f) * tex2d_linear_mipmap_linear(texcoords, texture, dudx, dudy, dvdx, dvdy);
};

template<typename T, glm::precision P = glm::defaultp>
glm::tvec3<T, P> tex2d(const glm::tvec2<T, P>& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy)
{
	// Todo: Change everything to GLM.
	cv::Vec3f ret = (1.0f / 255.0f) * tex2d_linear_mipmap_linear(cv::Vec2f(texcoords[0], texcoords[1]), texture, dudx, dudy, dvdx, dvdy);
	return glm::tvec3<T, P>(ret[0], ret[1], ret[2]);
};

inline cv::Vec3f tex2d_linear_mipmap_linear(const cv::Vec2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx, float dvdy)
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

inline cv::Vec3f tex2d_linear(const cv::Vec2f& imageTexCoord, unsigned char mipmap_index, const Texture& texture)
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

// Todo: Split this function into the general (core-part) and the texturing part.
// Then, utils::extractTexture can re-use the core-part.
// Note: Maybe a bit outdated "todo" above.
inline boost::optional<TriangleToRasterize> process_prospective_tri(Vertex<float> v0, Vertex<float> v1, Vertex<float> v2, int viewport_width, int viewport_height, bool enable_backface_culling)
{
	using cv::Vec2f;
	using cv::Vec3f;
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
	Vec2f v0_screen = eos::render::clip_to_screen_space(Vec2f(t.v0.position[0], t.v0.position[1]), viewport_width, viewport_height);
	t.v0.position[0] = v0_screen[0];
	t.v0.position[1] = v0_screen[1];
	Vec2f v1_screen = clip_to_screen_space(Vec2f(t.v1.position[0], t.v1.position[1]), viewport_width, viewport_height);
	t.v1.position[0] = v1_screen[0];
	t.v1.position[1] = v1_screen[1];
	Vec2f v2_screen = clip_to_screen_space(Vec2f(t.v2.position[0], t.v2.position[1]), viewport_width, viewport_height);
	t.v2.position[0] = v2_screen[0];
	t.v2.position[1] = v2_screen[1];

	if (enable_backface_culling) {
		if (!are_vertices_ccw_in_screen_space(glm::tvec2<float>(t.v0.position), glm::tvec2<float>(t.v1.position), glm::tvec2<float>(t.v2.position)))
			return boost::none;
	}

	// Get the bounding box of the triangle:
	cv::Rect boundingBox = calculate_clipped_bounding_box(glm::tvec2<float>(t.v0.position), glm::tvec2<float>(t.v1.position), glm::tvec2<float>(t.v2.position), viewport_width, viewport_height);
	t.min_x = boundingBox.x;
	t.max_x = boundingBox.x + boundingBox.width;
	t.min_y = boundingBox.y;
	t.max_y = boundingBox.y + boundingBox.height;

	if (t.max_x <= t.min_x || t.max_y <= t.min_y) 	// Note: Can the width/height of the bbox be negative? Maybe we only need to check for equality here?
		return boost::none;

	// Which of these is for texturing, mipmapping, what for perspective?
	// for partial derivatives computation
	t.alphaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcoords[0] * t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcoords[0] * t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcoords[0] * t.one_over_z2));
	t.betaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.v0.texcoords[1] * t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.v1.texcoords[1] * t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.v2.texcoords[1] * t.one_over_z2));
	t.gammaPlane = plane(Vec3f(t.v0.position[0], t.v0.position[1], t.one_over_z0),
		Vec3f(t.v1.position[0], t.v1.position[1], t.one_over_z1),
		Vec3f(t.v2.position[0], t.v2.position[1], t.one_over_z2));
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

inline void raster_triangle(TriangleToRasterize triangle, cv::Mat colourbuffer, cv::Mat depthbuffer, boost::optional<Texture> texture, bool enable_far_clipping)
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
				if (z_affine < depthbuffer.at<double>(pixel_index_row, pixel_index_col) && draw)
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
