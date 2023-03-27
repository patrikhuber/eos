/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/FragmentShader.hpp
 *
 * Copyright 2017, 2023 Patrik Huber
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

#ifndef EOS_FRAGMENT_SHADER_HPP
#define EOS_FRAGMENT_SHADER_HPP

#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/texturing.hpp"
#include "eos/cpp17/optional.hpp"

#include "Eigen/Core"

// Fragment shaders are a more accurate name for the same functionality as Pixel shaders. They aren't pixels
// yet, since the output still has to past several tests (depth, alpha, stencil) as well as the fact that one
// may be using anti-aliasing, which renders one-fragment-to-one-pixel non-true.
// The "pixel" in "pixel shader" is a misnomer because the pixel shader doesn't operate on pixels directly.
// The pixel shader operates on "fragments" which may or may not end up as actual pixels, depending on several
// factors outside of the pixel shader.
// The shaders *can not* depend on any state - they have to be able to run independently and in parallel!
// But can they really? What about the z-test? It happens earlier at the moment - which is good?

namespace eos {
namespace render {

/**
 * @brief A simple fragment shader that does vertex-colouring.
 *
 * Uses the vertex colour data to shade the given fragment / pixel location.
 */
class VertexColoringFragmentShader
{
public:
    /**
     * @brief Todo.
     *
     * Todo.
     * lambda is not perspectively corrected. Note: In our case, it is, as we do it in the raster loop at
     * the moment?
     * The given colour values should be in the range [0, 1]. The returned colour will also be in the range
     * [0, 1]. Note/Todo: Can colour values be <0 and >1? If so, we should document and tell the user to
     * perhaps clamp the values.
     *
     * @param[in] x X.
     * @return RGBA colour in the range [0, 1].
     */
    template <typename T>
    Eigen::Vector4<T> shade_triangle_pixel(int x, int y, const detail::Vertex<T>& point_a,
                                           const detail::Vertex<T>& point_b, const detail::Vertex<T>& point_c,
                                           const Eigen::Vector3<T>& lambda,
                                           const cpp17::optional<Texture>& texture, float dudx, float dudy,
                                           float dvdx, float dvdy)
    {
        // attributes interpolation
        const Eigen::Vector3<T> color_persp =
            lambda[0] * point_a.color + lambda[1] * point_b.color + lambda[2] * point_c.color;
        return Eigen::Vector4<T>(color_persp[0], color_persp[1], color_persp[2], T(1));
    };
};

/**
 * @brief A fragment shader that textures.
 */
class TexturingFragmentShader
{
public:
    /**
     * @brief Todo.
     *
     * See comments above about lambda (persp. corrected?).
     *
     * This shader reads the colour value from the given \p texture.
     * The returned colour will be in the range [0, 1].
     * Note/Todo: Can colour values be <0 and >1? If so, we should document and tell the user to perhaps clamp
     * the values.
     *
     * @param[in] x X.
     * @return RGBA colour in the range [0, 1].
     */
    template <typename T>
    Eigen::Vector4<T> shade_triangle_pixel(int x, int y, const detail::Vertex<T>& point_a,
                                           const detail::Vertex<T>& point_b, const detail::Vertex<T>& point_c,
                                           const Eigen::Vector3<T>& lambda,
                                           const cpp17::optional<Texture>& texture, float dudx, float dudy,
                                           float dvdx, float dvdy)
    {
        Eigen::Vector2<T> texcoords_persp =
            lambda[0] * point_a.texcoords + lambda[1] * point_b.texcoords + lambda[2] * point_c.texcoords;

        // The Texture is in BGR, thus tex2D returns BGR
        // Todo: Think about changing that.
        // tex2d divides the colour values by 255, so that the return value we get here is in the range [0, 1].
        const Eigen::Vector3<T> texture_color =
            detail::tex2d(texcoords_persp, texture.value(), dudx, dudy, dvdx, dvdy); // uses the current texture
        const Eigen::Vector3<T> pixel_color =
            Eigen::Vector3<T>(texture_color[2], texture_color[1], texture_color[0]);
        // other: color.mul(tex2D(texture, texCoord));
        return Eigen::Vector4<T>(pixel_color[0], pixel_color[1], pixel_color[2], T(1));
    };
};

/**
 * @brief Computes inverse perspectively correct lambda.
 *
 * X.
 * Inverts the perspective texture mapping. Can be derived using some tedious algebra.
 * Todo: Probably move to a texturing file, internal/detail one, where we will also put the tex2d, mipmapping
 * etc stuff?
 *
 * @param[in] X X.
 * @return X.
 */
template <typename T>
Eigen::Vector3<T> compute_inverse_perspectively_correct_lambda(const Eigen::Vector3<T>& lambda_world,
                                                               const T& one_over_w0, const T& one_over_w1,
                                                               const T& one_over_w2)
{
    const float w0 = 1 / one_over_w0;
    const float w1 = 1 / one_over_w1;
    const float w2 = 1 / one_over_w2;

    const float d = w0 - (w0 - w1) * lambda_world.y() - (w0 - w2) * lambda_world.z();
    if (d == 0)
        return lambda_world;

    Eigen::Vector3<T> lambda;

    lambda.z() = lambda_world.z() * w2 / d;
    lambda.y() = lambda_world.y() * w1 / d;

    lambda.x() = 1 - lambda.y() - lambda.z();
    return lambda;
};

/**
 * @brief A fragment shader that is used to extract, or remap, texture from an image to a UV map (i.e. the
 * reverse process of texturing).
 */
class ExtractionFragmentShader
{
public:
    /**
     * @brief Todo.
     *
     * Todo.
     * Inverts the perspective texture mapping. Can be derived using some tedious algebra.
     * NOTE: This one actually takes/needs the perspectively corrected lambda I think!
     *
     * Todo: Probably move to a texturing file, internal/detail one, where we will also put the tex2d,
     * mipmapping etc stuff?
     *
     * @param[in] X X.
     * @return X.
     */
    template <typename T>
    Eigen::Vector4<T> shade_triangle_pixel(int x, int y, const detail::Vertex<T>& point_a,
                                           const detail::Vertex<T>& point_b, const detail::Vertex<T>& point_c,
                                           const Eigen::Vector3<T>& lambda,
                                           const cpp17::optional<Texture>& texture, float dudx, float dudy,
                                           float dvdx, float dvdy)
    {
        const auto corrected_lambda = compute_inverse_perspectively_correct_lambda(
            lambda, point_a.position.w(), point_b.position.w(), point_c.position.w());
        const Eigen::Vector2<T> texcoords_persp = corrected_lambda[0] * point_a.texcoords +
                                                  corrected_lambda[1] * point_b.texcoords +
                                                  corrected_lambda[2] * point_c.texcoords;

        // Texturing, no mipmapping:
        Eigen::Vector2<T> image_tex_coords = detail::texcoord_wrap(texcoords_persp);
        image_tex_coords[0] *= texture->mipmaps[0].width();
        image_tex_coords[1] *= texture->mipmaps[0].height();
        const Eigen::Vector3<T> texture_color =
            detail::tex2d_linear(image_tex_coords, 0, texture.value()) / 255.0f;
        const Eigen::Vector3<T> pixel_color =
            Eigen::Vector3<T>(texture_color[2], texture_color[1], texture_color[0]);
        return Eigen::Vector4<T>(pixel_color[0], pixel_color[1], pixel_color[2], T(1));
    };
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_FRAGMENT_SHADER_HPP */
