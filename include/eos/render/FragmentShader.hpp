/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/FragmentShader.hpp
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

#ifndef EOS_FRAGMENT_SHADER_HPP
#define EOS_FRAGMENT_SHADER_HPP

#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/texturing.hpp"
#include "eos/cpp17/optional.hpp"

#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

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
    template <typename T, glm::precision P = glm::defaultp>
    glm::tvec4<T, P> shade_triangle_pixel(int x, int y, const detail::Vertex<T, P>& point_a,
                                          const detail::Vertex<T, P>& point_b,
                                          const detail::Vertex<T, P>& point_c, const glm::tvec3<T, P>& lambda,
                                          const cpp17::optional<Texture>& texture, float dudx, float dudy,
                                          float dvdx, float dvdy)
    {
        // attributes interpolation
        glm::tvec3<T, P> color_persp =
            lambda[0] * point_a.color + lambda[1] * point_b.color + lambda[2] * point_c.color;
        return glm::tvec4<T, P>(color_persp, T(1));
    };
};

/**
 * @brief A fragment shader that textures...
 *
 * X.
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
    template <typename T, glm::precision P = glm::defaultp>
    glm::tvec4<T, P> shade_triangle_pixel(int x, int y, const detail::Vertex<T, P>& point_a,
                                          const detail::Vertex<T, P>& point_b,
                                          const detail::Vertex<T, P>& point_c, const glm::tvec3<T, P>& lambda,
                                          const cpp17::optional<Texture>& texture, float dudx,
                                          float dudy, float dvdx, float dvdy)
    {
        glm::tvec2<T, P> texcoords_persp =
            lambda[0] * point_a.texcoords + lambda[1] * point_b.texcoords + lambda[2] * point_c.texcoords;

        // The Texture is in BGR, thus tex2D returns BGR
        // Todo: Think about changing that.
        // tex2d divides the colour values by 255, so that the return value we get here is in the range [0, 1].
        glm::tvec3<T, P> texture_color =
            detail::tex2d(texcoords_persp, texture.value(), dudx, dudy, dvdx, dvdy); // uses the current texture
        glm::tvec3<T, P> pixel_color = glm::tvec3<T, P>(texture_color[2], texture_color[1], texture_color[0]);
        // other: color.mul(tex2D(texture, texCoord));
        return glm::tvec4<T, P>(pixel_color, T(1));
    };
};

/**
 * @brief X.
 *
 * X.
 * Inverts the perspective texture mapping. Can be derived using some tedious algebra.
 * Todo: Probably move to a texturing file, internal/detail one, where we will also put the tex2d, mipmapping
 * etc stuff?
 *
 * @param[in] X X.
 * @return X.
 */
template <typename T, glm::precision P = glm::defaultp>
glm::tvec3<T, P> compute_inverse_perspectively_correct_lambda(const glm::tvec3<T, P>& lambda_world,
                                                              const T& one_over_w0, const T& one_over_w1,
                                                              const T& one_over_w2)
{
    float w0 = 1 / one_over_w0;
    float w1 = 1 / one_over_w1;
    float w2 = 1 / one_over_w2;

    float d = w0 - (w0 - w1) * lambda_world.y - (w0 - w2) * lambda_world.z;
    if (d == 0)
        return lambda_world;

    glm::tvec3<T, P> lambda;

    lambda.z = lambda_world.z * w2 / d;
    lambda.y = lambda_world.y * w1 / d;

    lambda.x = 1 - lambda.y - lambda.z;
    return lambda;
};

class ExtractionFragmentShader
{
public:
    /**
     * @brief X.
     *
     * X.
     * Inverts the perspective texture mapping. Can be derived using some tedious algebra.
     * NOTE: This one actually takes/needs the perspectively corrected lambda I think!
     *
     * Todo: Probably move to a texturing file, internal/detail one, where we will also put the tex2d,
     * mipmapping etc stuff?
     *
     * @param[in] X X.
     * @return X.
     */
    template <typename T, glm::precision P = glm::defaultp>
    glm::tvec4<T, P> shade_triangle_pixel(int x, int y, const detail::Vertex<T, P>& point_a,
                                          const detail::Vertex<T, P>& point_b,
                                          const detail::Vertex<T, P>& point_c, const glm::tvec3<T, P>& lambda,
                                          const cpp17::optional<Texture>& texture, float dudx, float dudy,
                                          float dvdx, float dvdy)
    {
        auto corrected_lambda = compute_inverse_perspectively_correct_lambda(
            lambda, point_a.position.w, point_b.position.w, point_c.position.w);
        glm::tvec2<T, P> texcoords_persp = corrected_lambda[0] * point_a.texcoords +
                                           corrected_lambda[1] * point_b.texcoords +
                                           corrected_lambda[2] * point_c.texcoords;

        // Texturing, no mipmapping:
        glm::vec2 image_tex_coords = detail::texcoord_wrap(texcoords_persp);
        image_tex_coords[0] *= texture->mipmaps[0].width();
        image_tex_coords[1] *= texture->mipmaps[0].height();
        glm::vec3 texture_color = detail::tex2d_linear(image_tex_coords, 0, texture.value()) / 255.0f;
        glm::tvec3<T, P> pixel_color = glm::tvec3<T, P>(texture_color[2], texture_color[1], texture_color[0]);
        return glm::tvec4<T, P>(pixel_color, T(1));
    };
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_FRAGMENT_SHADER_HPP */
