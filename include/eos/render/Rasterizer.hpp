/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/Rasterizer.hpp
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

#ifndef EOS_RASTERIZER_HPP
#define EOS_RASTERIZER_HPP

#include "eos/core/Rect.hpp"
#include "eos/core/Image.hpp"
#include "eos/core/image/utils.hpp"
#include "eos/render/Texture.hpp"
#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/plane.hpp"
#include "eos/render/detail/utils.hpp"
#include "eos/cpp17/optional.hpp"

#include <limits>

namespace eos {
namespace render {

/**
 * @brief Todo.
 *
 * X.
 *
 * @tparam FragmentShaderType X.
 */
template <typename FragmentShaderType>
class Rasterizer
{
public:
    Rasterizer(int viewport_width, int viewport_height)
        : viewport_width(viewport_width), viewport_height(viewport_height)
    {
        clear_buffers();
    };

    /**
     * @brief Todo.
     *
     * X
     *
     * @param[in] vertex X.
     * @ return X.
     */
    template <typename T, glm::precision P = glm::defaultp>
    void raster_triangle(const detail::Vertex<T, P>& point_a, const detail::Vertex<T, P>& point_b,
                         const detail::Vertex<T, P>& point_c, const cpp17::optional<Texture>& texture)
    {
        // We already calculated this in the culling/clipping stage. Maybe we should save/cache it after all.
        const auto boundingBox = detail::calculate_clipped_bounding_box(
            glm::tvec2<T, P>(point_a.position.x, point_a.position.y),
            glm::tvec2<T, P>(point_b.position.x, point_b.position.y),
            glm::tvec2<T, P>(point_c.position.x, point_c.position.y), viewport_width, viewport_height);
        const auto min_x = boundingBox.x;
        const auto max_x = boundingBox.x + boundingBox.width;
        const auto min_y = boundingBox.y;
        const auto max_y = boundingBox.y + boundingBox.height;

        // These are triangle-specific, i.e. calculate once per triangle.
        // These ones are needed for perspective correct lambdas! (as well as mipmapping)
        const auto& one_over_w0 = point_a.position[3];
        const auto& one_over_w1 = point_b.position[3];
        const auto& one_over_w2 = point_c.position[3];

        // These are triangle-specific, i.e. calculate once per triangle.
        // For partial derivatives computation (for mipmapping, texturing) (they work on screen-space coords):
        using eos::render::detail::plane;
        const auto alpha_plane = plane(
            glm::tvec3<T, P>(point_a.position[0], point_a.position[1], point_a.texcoords[0] * one_over_w0),
            glm::tvec3<T, P>(point_b.position[0], point_b.position[1], point_b.texcoords[0] * one_over_w1),
            glm::tvec3<T, P>(point_c.position[0], point_c.position[1], point_c.texcoords[0] * one_over_w2));
        const auto beta_plane = plane(
            glm::tvec3<T, P>(point_a.position[0], point_a.position[1], point_a.texcoords[1] * one_over_w0),
            glm::tvec3<T, P>(point_b.position[0], point_b.position[1], point_b.texcoords[1] * one_over_w1),
            glm::tvec3<T, P>(point_c.position[0], point_c.position[1], point_c.texcoords[1] * one_over_w2));
        const auto gamma_plane =
            plane(glm::tvec3<T, P>(point_a.position[0], point_a.position[1], one_over_w0),
                  glm::tvec3<T, P>(point_b.position[0], point_b.position[1], one_over_w1),
                  glm::tvec3<T, P>(point_c.position[0], point_c.position[1], one_over_w2));
        const auto one_over_alpha_c = 1.0f / alpha_plane.c;
        const auto one_over_beta_c = 1.0f / beta_plane.c;
        const auto one_over_gamma_c = 1.0f / gamma_plane.c;
        const auto alpha_ffx = -alpha_plane.a * one_over_alpha_c;
        const auto beta_ffx = -beta_plane.a * one_over_beta_c;
        const auto gamma_ffx = -gamma_plane.a * one_over_gamma_c;
        const auto alpha_ffy = -alpha_plane.b * one_over_alpha_c;
        const auto beta_ffy = -beta_plane.b * one_over_beta_c;
        const auto gamma_ffy = -gamma_plane.b * one_over_gamma_c;

        for (int yi = min_y; yi <= max_y; ++yi)
        {
            for (int xi = min_x; xi <= max_x; ++xi)
            {
                // we want centers of pixels to be used in computations. Todo: Do we? Do we pass it with or
                // without +0.5 to the FragShader?
                const float x = static_cast<float>(xi) + 0.5f; // double? T?
                const float y = static_cast<float>(yi) + 0.5f;

                // These will be used for barycentric weights computation
                using detail::implicit_line;
                const double one_over_v0ToLine12 =
                    1.0 / implicit_line(point_a.position[0], point_a.position[1], point_b.position,
                                        point_c.position);
                const double one_over_v1ToLine20 =
                    1.0 / implicit_line(point_b.position[0], point_b.position[1], point_c.position,
                                        point_a.position);
                const double one_over_v2ToLine01 =
                    1.0 / implicit_line(point_c.position[0], point_c.position[1], point_a.position,
                                        point_b.position);
                // Affine barycentric weights:
                double alpha = implicit_line(x, y, point_b.position, point_c.position) * one_over_v0ToLine12;
                double beta = implicit_line(x, y, point_c.position, point_a.position) * one_over_v1ToLine20;
                double gamma = implicit_line(x, y, point_a.position, point_b.position) * one_over_v2ToLine01;

                // if pixel (x, y) is inside the triangle or on one of its edges
                if (alpha >= 0 && beta >= 0 && gamma >= 0)
                {
                    const int pixel_index_row = yi;
                    const int pixel_index_col = xi;

                    // TODO: Check this one. What about perspective?
                    const double z_affine = alpha * static_cast<double>(point_a.position[2]) +
                                            beta * static_cast<double>(point_b.position[2]) +
                                            gamma * static_cast<double>(point_c.position[2]);

                    bool draw = true;
                    if (enable_far_clipping)
                    {
                        if (z_affine > 1.0)
                        {
                            draw = false;
                        }
                    }

                    bool passes_depth_test = false;
                    if (enable_depth_test)
                    {
                        // If enable_depth_test=false, avoid accessing the depthbuffer at all - it might be
                        // empty or have other dimensions.
                        passes_depth_test =
                            (z_affine < depthbuffer(pixel_index_row, pixel_index_col));
                    }
                    // The '<= 1.0' clips against the far-plane in NDC. We clip against the near-plane
                    // earlier.
                    // if (z_affine < depthbuffer.at<double>(pixelIndexRow, pixelIndexCol)/* && z_affine <=
                    // 1.0*/) // what to do in ortho case without n/f "squashing"? should we always squash? or
                    // a flag?
                    if ((passes_depth_test && draw) || enable_depth_test == false)
                    {
                        // perspective-correct barycentric weights
                        // Todo: Check this in the original/older implementation, i.e. if all is still
                        // perspective-correct. I think so. Also compare 1:1 with OpenGL.
                        if (!extracting_tex) // Pass the uncorrected lambda if we're extracting tex... hack...
                                             // do properly!
                        {
                            double d = alpha * one_over_w0 + beta * one_over_w1 + gamma * one_over_w2;
                            d = 1.0 / d;
                            alpha *= d * one_over_w0; // In case of affine cam matrix, everything is 1 and
                                                      // a/b/g don't get changed.
                            beta *= d * one_over_w1;
                            gamma *= d * one_over_w2;
                        }
                        glm::tvec3<T, P> lambda(alpha, beta, gamma);

                        glm::tvec4<T, P> pixel_color;
                        if (texture)
                        {
                            // check if texture != NULL?
                            // partial derivatives (for mip-mapping, not needed for texturing otherwise!)
                            const float u_over_z =
                                -(alpha_plane.a * x + alpha_plane.b * y + alpha_plane.d) * one_over_alpha_c;
                            const float v_over_z =
                                -(beta_plane.a * x + beta_plane.b * y + beta_plane.d) * one_over_beta_c;
                            const float one_over_z =
                                -(gamma_plane.a * x + gamma_plane.b * y + gamma_plane.d) * one_over_gamma_c;
                            const float one_over_squared_one_over_z = 1.0f / std::pow(one_over_z, 2);

                            // partial derivatives of U/V coordinates with respect to X/Y pixel's screen
                            // coordinates
                            // These are exclusively used for the mipmap level computation (i.e. which mipmap
                            // levels to use).
                            // They're not needed for texturing otherwise at all!
                            float dudx =
                                one_over_squared_one_over_z * (alpha_ffx * one_over_z - u_over_z * gamma_ffx);
                            float dudy =
                                one_over_squared_one_over_z * (beta_ffx * one_over_z - v_over_z * gamma_ffx);
                            float dvdx =
                                one_over_squared_one_over_z * (alpha_ffy * one_over_z - u_over_z * gamma_ffy);
                            float dvdy =
                                one_over_squared_one_over_z * (beta_ffy * one_over_z - v_over_z * gamma_ffy);
                            dudx *= texture.value().mipmaps[0].width();
                            dudy *= texture.value().mipmaps[0].width();
                            dvdx *= texture.value().mipmaps[0].height();
                            dvdy *= texture.value().mipmaps[0].height();

                            // Why does it need x and y? Maybe some shaders (eg TexExtr?) need it?
                            pixel_color = fragment_shader.shade_triangle_pixel(
                                x, y, point_a, point_b, point_c, lambda, texture, dudx, dudy, dvdx, dvdy);

                        } else
                        { // We use vertex-coloring
                            // Why does it need x and y?
                            pixel_color = fragment_shader.shade_triangle_pixel(
                                x, y, point_a, point_b, point_c, lambda, texture, 0, 0, 0, 0);
                        }

                        // clamp bytes to 255
                        // Todo: Proper casting (rounding?)? And we don't clamp/max against 255? Use
                        // glm::clamp?
                        const unsigned char red =
                            static_cast<unsigned char>(255.0f * std::min(pixel_color[0], T(1)));
                        const unsigned char green =
                            static_cast<unsigned char>(255.0f * std::min(pixel_color[1], T(1)));
                        const unsigned char blue =
                            static_cast<unsigned char>(255.0f * std::min(pixel_color[2], T(1)));
                        const unsigned char alpha =
                            static_cast<unsigned char>(255.0f * std::min(pixel_color[3], T(1)));

                        // update buffers
                        colorbuffer(pixel_index_row, pixel_index_col)[0] = blue;
                        colorbuffer(pixel_index_row, pixel_index_col)[1] = green;
                        colorbuffer(pixel_index_row, pixel_index_col)[2] = red;
                        colorbuffer(pixel_index_row, pixel_index_col)[3] = alpha;
                        if (enable_depth_test) // TODO: A better name for this might be enable_zbuffer? or
                                               // enable_zbuffer_test?
                        {
                            depthbuffer(pixel_index_row, pixel_index_col) = z_affine;
                        }
                    }
                }
            }
        }
    };

    /**
     * @brief Resets the colour and depth buffers.
     *
     * Sets the colour buffer to all white with zeros for the alpha channel, and the depth buffer to
     * std::numeric_limits<double>::max().
     *
     * If multiple images are rendered/rasterised, then this function can be called before rendering a new
     * image, to clear the colour and depth buffers.
     */
    void clear_buffers()
    {
        colorbuffer = eos::core::image::constant(viewport_height, viewport_width,
                                                 eos::core::Pixel<std::uint8_t, 4>(255, 255, 255, 0));
        depthbuffer =
            eos::core::image::constant(viewport_height, viewport_width, std::numeric_limits<double>::max());
    };

private:
    FragmentShaderType fragment_shader;

public:                            // will eventually go private
    bool enable_depth_test = true; // maybe get rid of this again, it was just as a hack.
    bool extracting_tex = false;
    bool enable_far_clipping = true;

    int viewport_width;
    int viewport_height;

    eos::core::Image4u colorbuffer;
	eos::core::Image1d depthbuffer;
};

} // namespace render
} // namespace eos

#endif /* EOS_RASTERIZER_HPP */
