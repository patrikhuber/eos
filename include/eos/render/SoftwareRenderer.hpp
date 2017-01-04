/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/SoftwareRenderer.hpp
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

#ifndef SOFTWARERENDERER_HPP_
#define SOFTWARERENDERER_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/Vertex.hpp"
#include "eos/render/utils.hpp" // for Texture, potentially others

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec4.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <array>
#include <limits>

/**
 * @file include/eos/render/SoftwareRenderer.hpp
 * @brief This file implements a software renderer, in the spirit of OpenGL conventions and vertex and
 * fragment shaders.
 *
 * Might be worth adding the comments from render.hpp, regarding the pipeline and
 * OpenGL conventions etc.
 */

namespace eos {
namespace render {

// Forward declarations (these functions should probably be moved into detail/):
template <typename T, glm::precision P = glm::defaultp>
glm::tvec4<T, P> divide_by_w(const glm::tvec4<T, P>& vertex);

template <typename T, glm::precision P = glm::defaultp>
std::vector<detail::v2::Vertex<T, P>>
clip_polygon_to_plane_in_4d(const std::vector<detail::v2::Vertex<T, P>>& vertices,
                            const glm::tvec4<T, P>& plane_normal);

/**
 * @brief X.
 *
 * Can this go into the SoftwareRenderer class or something? No, I think FragShader needs it? Where to put it?
 */
template <typename T, glm::precision P = glm::defaultp>
using Triangle = std::array<detail::v2::Vertex<T, P>, 3>;

/**
 * @brief X.
 *
 * Longer.
 *
 * @tparam VertexShaderType vs-type.
 * @tparam FragmentShaderType fs-type.
 */
template <typename VertexShaderType, typename FragmentShaderType>
class SoftwareRenderer
{
public:
    SoftwareRenderer(int viewport_width, int viewport_height)
        : viewport_width(viewport_width), viewport_height(viewport_height)
    {
        colorbuffer = cv::Mat(viewport_height, viewport_width, CV_8UC4, cv::Scalar::all(255));
        depthbuffer =
            std::numeric_limits<double>::max() * Mat::ones(viewport_height, viewport_width, CV_64FC1);
    };

    // Deleting copy constructor and assignment for now because e.g. the framebuffer member is a
    // cv::Mat, so copying a renderer will not copy the framebuffer. We may want to fix that properly.
    SoftwareRenderer(const SoftwareRenderer& rhs) = delete;
    SoftwareRenderer& operator=(const SoftwareRenderer&) = delete;

    /**
     * @brief Todo.
     *
     * Todo.
     * The returned framebuffer cv::Mat is a smart-pointer to the colorbuffer object inside SoftwareRenderer,
     * and will be overwritten on the next call to render(). If you want a copy, use .clone()!
     *
     * @param[in] mesh The mesh to render.
     * @param[in] model_view_matrix The mesh to render.
     * @param[in] projection_matrix The mesh to render.
     * @param[in] texture The mesh to render.
     * @ return The framebuffer (colourbuffer) with the rendered object.
     */
    template <typename T, glm::precision P = glm::defaultp>
    cv::Mat render(const eos::core::Mesh& mesh, const glm::tmat4x4<T, P>& model_view_matrix,
                   const glm::tmat4x4<T, P>& projection_matrix,
                   const boost::optional<eos::render::Texture>& texture = boost::none)
    {
        assert(mesh.vertices.size() == mesh.colors.size() ||
               mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or,
                                     // alternatively, it has to be a shape-only model.
        assert(mesh.vertices.size() == mesh.texcoords.size() ||
               mesh.texcoords.empty()); // same for the texcoords
        // Add another assert: If cv::Mat texture != empty (and/or texturing=true?), then we need texcoords?

        using cv::Mat;
        using std::vector;

        vector<glm::tvec4<T, P>> clipspace_vertices;
        for (const auto& vertex_position : mesh.vertices)
        {
            clipspace_vertices.push_back(
                vertex_shader(vertex_position, model_view_matrix, projection_matrix));
            // Note: if mesh.colors.empty() (in case of shape-only model!), then the vertex colour is no
            // longer set to gray. But we don't want that here, maybe we only want texturing, then we don't
            // need vertex-colours at all! We can do it in a custom VertexShader if needed!
        }

        // All vertices are in clip-space now. Prepare the rasterisation stage:
        vector<Triangle<T, P>> triangles_to_raster;
        // This builds the (one and final) triangles to render. Meaning: The triangles formed of mesh.tvi (the
        // ones that survived the clip/culling), plus possibly more that intersect one of the frustum planes
        // (i.e. this can generate new triangles with new pos/vc/texcoords).
        for (const auto& tri_indices : mesh.tvi)
        {
            unsigned char visibility_bits[3];
            for (unsigned char k = 0; k < 3; k++)
            {
                visibility_bits[k] = 0;
                const auto x_cc = clipspace_vertices[tri_indices[k]].x;
                const auto y_cc = clipspace_vertices[tri_indices[k]].y;
                const auto z_cc = clipspace_vertices[tri_indices[k]].z;
                const auto w_cc = clipspace_vertices[tri_indices[k]].w;
                if (x_cc < -w_cc) // true if outside of view frustum. False if on or inside the plane.
                    visibility_bits[k] |= 1; // set bit if outside of frustum
                if (x_cc > w_cc)
                    visibility_bits[k] |= 2;
                if (y_cc < -w_cc)
                    visibility_bits[k] |= 4;
                if (y_cc > w_cc)
                    visibility_bits[k] |= 8;
                if (enable_near_clipping && z_cc < -w_cc) // near plane frustum clipping
                    visibility_bits[k] |= 16;
                if (enable_far_clipping && z_cc > w_cc) // far plane frustum clipping
                    visibility_bits[k] |= 32;
            } // if all bits are 0, then it's inside the frustum
            // all vertices are not visible - reject the triangle.
            if ((visibility_bits[0] & visibility_bits[1] & visibility_bits[2]) > 0)
            {
                continue;
            }
            // all vertices are visible - pass the whole triangle to the rasteriser. = All bits of all 3
            // triangles are 0.
            if ((visibility_bits[0] | visibility_bits[1] | visibility_bits[2]) == 0)
            {
                // relevant part of process_prospective_tri:
                std::array<glm::tvec4<T, P>, 3> prospective_tri{
                    divide_by_w(clipspace_vertices[tri_indices[0]]),
                    divide_by_w(clipspace_vertices[tri_indices[1]]),
                    divide_by_w(clipspace_vertices[tri_indices[2]])};
                // We have a prospective tri in NDC coords now, with its vertices having coords [x_ndc, y_ndc,
                // z_ndc, 1/w_clip].

                // Replaces x and y of the NDC coords with the screen coords. Keep z and w the same.
                const glm::tvec2<T, P> v0_screen = clip_to_screen_space(
                    prospective_tri[0].x, prospective_tri[0].y, viewport_width, viewport_height);
                prospective_tri[0].x = v0_screen.x;
                prospective_tri[0].y = v0_screen.y;
                const glm::tvec2<T, P> v1_screen = clip_to_screen_space(
                    prospective_tri[1].x, prospective_tri[1].y, viewport_width, viewport_height);
                prospective_tri[1].x = v1_screen.x;
                prospective_tri[1].y = v1_screen.y;
                const glm::tvec2<T, P> v2_screen = clip_to_screen_space(
                    prospective_tri[2].x, prospective_tri[2].y, viewport_width, viewport_height);
                prospective_tri[2].x = v2_screen.x;
                prospective_tri[2].y = v2_screen.y;

                // Culling (front/back/none - or what are OpenGL's modes?). Do we do any culling
                // elsewhere? No?
                if (enable_backface_culling)
                {
                    if (!detail::are_vertices_ccw_in_screen_space(
                            glm::tvec2<T, P>(prospective_tri[0].x, prospective_tri[0].y),
                            glm::tvec2<T, P>(prospective_tri[1].x, prospective_tri[1].y),
                            glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y)))
                        continue;
                }

                // Get the bounding box of the triangle:
                const cv::Rect boundingBox = detail::calculate_clipped_bounding_box(
                    glm::tvec2<T, P>(prospective_tri[0].x, prospective_tri[0].y),
                    glm::tvec2<T, P>(prospective_tri[1].x, prospective_tri[1].y),
                    glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y), viewport_width,
                    viewport_height);
                const auto min_x = boundingBox.x;
                const auto max_x = boundingBox.x + boundingBox.width;
                const auto min_y = boundingBox.y;
                const auto max_y = boundingBox.y + boundingBox.height;
                if (max_x <= min_x || max_y <= min_y)
                { // Note: Can the width/height of the bbox be negative? Maybe we only need to check for
                    // equality here?
                    continue;
                }

                // If we're here, the triangle is CCW in screen space and the bbox is inside the viewport!
                triangles_to_raster.push_back(
                    Triangle<T, P>{detail::v2::Vertex<T, P>{prospective_tri[0], mesh.colors[tri_indices[0]],
                                                            mesh.texcoords[tri_indices[0]]},
                                   detail::v2::Vertex<T, P>{prospective_tri[1], mesh.colors[tri_indices[1]],
                                                            mesh.texcoords[tri_indices[1]]},
                                   detail::v2::Vertex<T, P>{prospective_tri[2], mesh.colors[tri_indices[2]],
                                                            mesh.texcoords[tri_indices[2]]}});
                continue; // Triangle was either added or not added. Continue with next triangle.
            }
            // At this point, the triangle is known to be intersecting one of the view frustum's planes
            // Note: It seems that this is only w.r.t. the near-plane. If a triangle is partially outside the
            // tlbr viewport, it'll get rejected.
            // Well, 'z' of these triangles seems to be -1, so is that really the near plane?
            std::vector<detail::v2::Vertex<T, P>> vertices;
            vertices.reserve(3);
            vertices.push_back(detail::v2::Vertex<T, P>{clipspace_vertices[tri_indices[0]],
                                                        mesh.colors[tri_indices[0]],
                                                        mesh.texcoords[tri_indices[0]]});
            vertices.push_back(detail::v2::Vertex<T, P>{clipspace_vertices[tri_indices[1]],
                                                        mesh.colors[tri_indices[1]],
                                                        mesh.texcoords[tri_indices[1]]});
            vertices.push_back(detail::v2::Vertex<T, P>{clipspace_vertices[tri_indices[2]],
                                                        mesh.colors[tri_indices[2]],
                                                        mesh.texcoords[tri_indices[2]]});
            // split the triangle if it intersects the near plane:
            if (enable_near_clipping)
            {
                vertices = clip_polygon_to_plane_in_4d(
                    vertices, glm::tvec4<T, P>(T(0.0), T(0.0), T(-1.0),
                                               T(-1.0))); // "Normal" (or "4D hyperplane") of the near-plane.
                                                          // I tested it and it works like this but I'm a
                                                          // little bit unsure because Songho says the normal
                                                          // of the near-plane is (0,0,-1,1) (maybe I have to
                                                          // switch around the < 0 checks in the function?)
            }

            // Triangulation of the polygon formed of the 'vertices' array:
            if (vertices.size() >= 3)
            {
                for (unsigned char k = 0; k < vertices.size() - 2; k++)
                {
                    // Build a triangle from vertices[0], vertices[1 + k], vertices[2 + k]:
                    // Add to triangles_to_raster if it passed culling etc.
                    // TODO: This does the same as above - the code is copied 1:1. Avoid!
                    // COPY START (but init of prospective_tri is changed, as well as init of 't'.)
                    std::array<glm::tvec4<T, P>, 3> prospective_tri{divide_by_w(vertices[0].position),
                                                                    divide_by_w(vertices[1 + k].position),
                                                                    divide_by_w(vertices[2 + k].position)};

                    const glm::tvec2<T, P> v0_screen = clip_to_screen_space(
                        prospective_tri[0].x, prospective_tri[0].y, viewport_width, viewport_height);
                    prospective_tri[0].x = v0_screen.x;
                    prospective_tri[0].y = v0_screen.y;
                    const glm::tvec2<T, P> v1_screen = clip_to_screen_space(
                        prospective_tri[1].x, prospective_tri[1].y, viewport_width, viewport_height);
                    prospective_tri[1].x = v1_screen.x;
                    prospective_tri[1].y = v1_screen.y;
                    const glm::tvec2<T, P> v2_screen = clip_to_screen_space(
                        prospective_tri[2].x, prospective_tri[2].y, viewport_width, viewport_height);
                    prospective_tri[2].x = v2_screen.x;
                    prospective_tri[2].y = v2_screen.y;

                    if (enable_backface_culling)
                    {
                        if (!detail::are_vertices_ccw_in_screen_space(
                                glm::tvec2<T, P>(prospective_tri[0].x, prospective_tri[0].y),
                                glm::tvec2<T, P>(prospective_tri[1].x, prospective_tri[1].y),
                                glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y)))
                            continue;
                    }

                    const cv::Rect boundingBox = detail::calculate_clipped_bounding_box(
                        glm::tvec2<T, P>(prospective_tri[0].x, prospective_tri[0].y),
                        glm::tvec2<T, P>(prospective_tri[1].x, prospective_tri[1].y),
                        glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y), viewport_width,
                        viewport_height);
                    const auto min_x = boundingBox.x;
                    const auto max_x = boundingBox.x + boundingBox.width;
                    const auto min_y = boundingBox.y;
                    const auto max_y = boundingBox.y + boundingBox.height;
                    if (max_x <= min_x || max_y <= min_y)
                    {
                        continue;
                    }

                    // If we're here, the triangle is CCW in screen space and the bbox is inside the viewport!
                    triangles_to_raster.push_back(
                        Triangle<T, P>{detail::v2::Vertex<T, P>{prospective_tri[0], vertices[0].color,
                                                                vertices[0].texcoords},
                                       detail::v2::Vertex<T, P>{prospective_tri[1], vertices[1 + k].color,
                                                                vertices[1 + k].texcoords},
                                       detail::v2::Vertex<T, P>{prospective_tri[2], vertices[2 + k].color,
                                                                vertices[2 + k].texcoords}});
                    // continue; // triangle was either added or not added. Continue with next triangle.
                    // COPY END
                }
            }

        } // end of loop over all triangles

        // Each triangle contains [x_screen, y_screen, z_ndc, 1/w_clip].
        // We may have more triangles than in the original mesh.

        // Raster each triangle and apply the fragment shader on each pixel:
        for (const auto& tri : triangles_to_raster)
        {
            raster_triangle(tri[0], tri[1], tri[2], texture);
        }
        return colorbuffer;
    };

public: // Todo: these should go private in the final implementation
    cv::Mat colorbuffer;
    cv::Mat depthbuffer;

    boost::optional<Texture> texture = boost::none;
    bool enable_backface_culling = false;
    bool enable_near_clipping = true;
    bool enable_far_clipping = true;
    bool enable_depth_test = true; // maybe get rid of this again, it was just as a hack.
    bool extracting_tex = false;

    VertexShaderType vertex_shader;
    FragmentShaderType fragment_shader; // Replace with Rasterizer, move FS into Rasterizer

    int viewport_width;
    int viewport_height;

public:
    /**
     * @brief Todo.
     *
     * X
     *
     * @param[in] vertex X.
     * @ return X.
     */
    template <typename T, glm::precision P = glm::defaultp>
    void raster_triangle(const detail::v2::Vertex<T, P>& point_a, const detail::v2::Vertex<T, P>& point_b,
                         const detail::v2::Vertex<T, P>& point_c, const boost::optional<Texture>& texture)
    {
        // We already calculated this in the culling/clipping stage. Maybe we should save/cache it after all.
        cv::Rect boundingBox = detail::calculate_clipped_bounding_box(
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
                            (z_affine < depthbuffer.at<double>(pixel_index_row, pixel_index_col));
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
                        double d = alpha * one_over_w0 + beta * one_over_w1 + gamma * one_over_w2;
                        d = 1.0 / d;
                        if (!extracting_tex) // Pass the uncorrected lambda if we're extracting tex... hack...
                                             // do properly!
                        {
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
                            dudx *= texture.get().mipmaps[0].cols;
                            dudy *= texture.get().mipmaps[0].cols;
                            dvdx *= texture.get().mipmaps[0].rows;
                            dvdy *= texture.get().mipmaps[0].rows;

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
                        colorbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[0] = blue;
                        colorbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[1] = green;
                        colorbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[2] = red;
                        colorbuffer.at<cv::Vec4b>(pixel_index_row, pixel_index_col)[3] = alpha;
                        if (enable_depth_test) // TODO: A better name for this might be enable_zbuffer? or
                                               // enable_zbuffer_test?
                        {
                            depthbuffer.at<double>(pixel_index_row, pixel_index_col) = z_affine;
                        }
                    }
                }
            }
        }
    };
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

/**
 * @brief Todo.
 *
 * This function copied from render_detail.hpp and adjusted for v2:: stuff.
 * I think it should go back to render_details eventually.
 *
 * @param[in] vertices X.
 * @param[in] plane_normal X.
 * @ return X.
 */
template <typename T, glm::precision P = glm::defaultp>
std::vector<detail::v2::Vertex<T, P>>
clip_polygon_to_plane_in_4d(const std::vector<detail::v2::Vertex<T, P>>& vertices,
                            const glm::tvec4<T, P>& plane_normal)
{
    std::vector<detail::v2::Vertex<T, P>> clipped_vertices;

    // We can have 2 cases:
    //	* 1 vertex visible: we make 1 new triangle out of the visible vertex plus the 2 intersection points
    // with the near-plane
    //  * 2 vertices visible: we have a quad, so we have to make 2 new triangles out of it.

    // See here for more info?
    // http://math.stackexchange.com/questions/400268/equation-for-a-line-through-a-plane-in-homogeneous-coordinates

    for (unsigned int i = 0; i < vertices.size(); i++)
    {
        int a = i;                         // the current vertex
        int b = (i + 1) % vertices.size(); // the following vertex (wraps around 0)

        using glm::dot;
        const T fa = dot(vertices[a].position, plane_normal); // Note: Shouldn't they be unit length?
        const T fb = dot(vertices[b].position,
                         plane_normal); // < 0 means on visible side, > 0 means on invisible side?

        if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) // one vertex is on the visible side of the plane, one
                                                      // on the invisible? so we need to split?
        {
            const glm::tvec4<T, P> direction = vertices[b].position - vertices[a].position;
            const T t = -(dot(plane_normal, vertices[a].position)) /
                        (dot(plane_normal, direction)); // the parametric value on the line, where the line to
                                                        // draw intersects the plane?

            // generate a new vertex at the line-plane intersection point
            const glm::tvec4<T, P> position = vertices[a].position + t * direction;
            const glm::tvec3<T, P> color = vertices[a].color + t * (vertices[b].color - vertices[a].color);
            const glm::tvec2<T, P> texcoords =
                vertices[a].texcoords +
                t * (vertices[b].texcoords -
                     vertices[a].texcoords); // We could omit that if we don't render with texture.

            if (fa < 0) // we keep the original vertex plus the new one
            {
                clipped_vertices.push_back(vertices[a]);
                clipped_vertices.push_back(detail::v2::Vertex<T, P>{position, color, texcoords});
            } else if (fb < 0) // we use only the new vertex
            {
                clipped_vertices.push_back(detail::v2::Vertex<T, P>{position, color, texcoords});
            }
        } else if (fa < 0 && fb < 0) // both are visible (on the "good" side of the plane), no splitting
                                     // required, use the current vertex
        {
            clipped_vertices.push_back(vertices[a]);
        }
        // else, both vertices are not visible, nothing to add and draw
    }

    return clipped_vertices;
};

} /* namespace render */
} /* namespace eos */

#endif /* SOFTWARERENDERER_HPP_ */
