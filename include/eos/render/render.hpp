/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/render.hpp
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

#ifndef RENDER_HPP_
#define RENDER_HPP_

#include "eos/core/Image.hpp"
#include "eos/core/image/utils.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/cpp17/optional.hpp"

#include <array>
#include <vector>

namespace eos {
namespace render {

/**
 * This file implements a software renderer conforming to OpenGL conventions. The
 * following are implementation notes, mostly for reference, or as a reminder of
 * what exactly is going on. Don't try to understand them :-)
 *
 * The renderer was initially based on code by Wojciech Sterna
 * (http://maxest.gct-game.net/content/vainmoinen/index.html), however, it has since
 * then been completely rewritten. Still I'd like to thank him for making his code
 * available and bravely answering my questions via email.
 *
 * Coordinate systems:
 * When specifying the vertices: +x = right, +y = up, we look into -z.
 * So z = 0.5 is in front of 0.0.
 * Z-Buffer:
 *
 * Shirley: Specify n and f with negative values. which makes sense b/c the points
 * are along the -z axis.
 * Consequences: notably: orthogonal(2, 3): Shirley has denominator (n-f).
 * In what space are the points in Shirley after this?
 * OGL: We're in the orthographic viewing volume looking down -z.
 * However, n and f are specified positive.

 * B/c the 3D points in front of the cam obviously still have negative z values, the
 * z-value is negated. So: n = 0.1, f = 100; With the given OpenGL ortho matrix,
 * it means a point on the near-plane which will have z = -0.1 will land up
 * on z_clip (which equals z_ndc with ortho because w=1) = -1, and a point on
 * the far plane z = -100 will have z_ndc = +1.
 *
 * That's also why in the perspective case, w_clip is set to -z_eye because
 * to project a point the formula is $x_p = (-n * x_e)/z_e$ (because our near is
 * specified with positive values, but the near-plane is _really_ at -n); but now we
 * just move the minus-sign to the denominator, $x_p = (n * x_e)/-z_e$, so in the projection matrix we can use
 * the (positive) n and f values and afterwards we divide by w = -z_e.
 *
 * http://www.songho.ca/opengl/gl_projectionmatrix.html
 *
 * Random notes:
 * clip-space: after applying the projection matrix.
 * ndc: after division by w
 * NDC cube: the range of x-coordinate from [l, r] to [-1, 1], the y-coordinate from [b, t] to [-1, 1] and the z-coordinate from [n, f] to [-1, 1].
 *
 * Note/Todo: I read that in screen space, OpenGL transform the z-values again to be between 0 and 1?
 *
 * In contrast to OGL, this renderer doesn't have state, it's just a function that gets called with all
 * necessary parameters. It's easiest for our purposes.
 *
 * Here's the whole rendering pipeline:
 * Model space
 * -> model transforms
 * World space
 * -> camera (view/eye) transform
 * View / eye / camera space ("truncated pyramid frustum". In case of ortho, it's already rectangular.)
 * -> perspective/ortho projection
 * Clip coords (x_c, y_c, z_c, w_c); the z-axis is flipped now. z [z=-n, z=-f] is mapped to [-1, +1] in case of ortho, but not yet in case of persp (it's also flipped though), but the not-[-1,1]-range is fine as we test against w_c. I.e. the larger the z-value, the further back we are.
 * Do frustum culling (clipping) here. Test the clip-coords with w_c, and discard if a tri is completely outside.
 * Of the partially visible tris, clip them against the near-plane and construct the visible part of the triangle.
 * We only do this for the near-plane here. Clipping to the near plane must be done here because after w-division triangles crossing it would get distorted.
 * "Then, OpenGL will reconstruct the edges of the polygon where clipping occurs."
 * -> Then divide by the w component of the clip coordinates
 * NDC. (now only 3D vectors: [x_ndc, y_ndc, z_ndc]). nearest points have z=-1, points on far plane have z=+1.
 * -> window transform. (also, OGL does some more to the z-buffer?)
 * Screen / window space
 * Directly after window-transform (still processing triangles), do backface culling with areVerticesCCWInScreenSpace()
 * Directly afterwards we calculate the triangle's bounding box and clip x/y (screen) against 0 and the viewport width/height.
 * Rasterising: Clipping against the far plane here by only drawing those pixels with a z-value of <= 1.0f.
 *
 * OGL: "both clipping (frustum culling) and NDC transformations are integrated into GL_PROJECTION matrix"
 *
 * Note: In both the ortho and persp case, points at z=-n end up at -1, z=-f at +1. In case of persp proj., this happens only after the divide by w.
 */

/**
 * Renders the given mesh onto a 2D image using 4x4 model-view and
 * projection matrices. Conforms to OpenGL conventions.
 *
 * @param[in] mesh A 3D mesh.
 * @param[in] model_view_matrix A 4x4 OpenGL model-view matrix.
 * @param[in] projection_matrix A 4x4 orthographic or perspective OpenGL projection matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] texture An optional texture map. If not given, vertex-colouring is used.
 * @param[in] enable_backface_culling Whether the renderer should perform backface culling. If true, only draw triangles with vertices ordered CCW in screen-space.
 * @param[in] enable_near_clipping Whether vertices should be clipped against the near plane.
 * @param[in] enable_far_clipping Whether vertices should be clipped against the far plane.
 * @return A pair with the colourbuffer as its first element and the depthbuffer as the second element.
 */
inline std::pair<core::Image4u, core::Image1d>
render(core::Mesh mesh, glm::tmat4x4<float> model_view_matrix, glm::tmat4x4<float> projection_matrix,
       int viewport_width, int viewport_height, const cpp17::optional<Texture>& texture = cpp17::nullopt,
       bool enable_backface_culling = false, bool enable_near_clipping = true,
       bool enable_far_clipping = true)
{
    // Some internal documentation / old todos or notes:
    // - maybe change and pass depthBuffer as an optional arg (&?), because usually we never need it outside
    //   the renderer. Or maybe even a getDepthBuffer().
    // - modelViewMatrix goes to eye-space (camera space), projection does ortho or perspective proj.
    // - bool enable_texturing = false; Maybe re-add later, not sure
    // - take a cv::Mat texture instead and convert to Texture internally? no, we don't want to recreate
    //   mipmap levels on each render() call.

    assert(mesh.vertices.size() == mesh.colors.size() ||
           mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or,
                                 // alternatively, it has to be a shape-only model.
    assert(mesh.vertices.size() == mesh.texcoords.size() || mesh.texcoords.empty()); // same for the texcoords
    // another assert: If cv::Mat texture != empty, then we need texcoords?

    using std::vector;

    core::Image4u colorbuffer =
        core::image::zeros<core::Pixel<std::uint8_t, 4>>(viewport_height, viewport_width);
    core::Image1d depthbuffer =
        core::image::constant(viewport_height, viewport_width, std::numeric_limits<double>::max());

    // Vertex shader:
    // processedVertex = shade(Vertex); // processedVertex : pos, col, tex, texweight
    // Assemble the vertices, project to clip space, and store as detail::Vertex (the internal
    // representation):
    vector<detail::Vertex<float>> clipspace_vertices;
    clipspace_vertices.reserve(mesh.vertices.size());
    for (int i = 0; i < mesh.vertices.size(); ++i)
    { // "previously": mesh.vertex
        const glm::tvec4<float> vertex(mesh.vertices[i][0], mesh.vertices[i][1], mesh.vertices[i][2], 1.0f);
        const glm::tvec4<float> clipspace_coords = projection_matrix * model_view_matrix * vertex;
        glm::tvec3<float> vertex_colour;
        if (mesh.colors.empty())
        {
            vertex_colour = glm::tvec3<float>(0.5f, 0.5f, 0.5f);
        } else
        {
            vertex_colour = glm::tvec3<float>(mesh.colors[i][0], mesh.colors[i][1], mesh.colors[i][2]);
        }
        clipspace_vertices.push_back(detail::Vertex<float>{
            clipspace_coords, vertex_colour, glm::tvec2<float>(mesh.texcoords[i][0], mesh.texcoords[i][1])});
    }

    // All vertices are in clip-space now.
    // Prepare the rasterisation stage.
    // For every vertex/tri:
    vector<detail::TriangleToRasterize> triangles_to_raster;
    for (const auto& tri_indices : mesh.tvi)
    {
        // Todo: Split this whole stuff up. Make a "clip" function, ... rename "processProspective..".. what
        // is "process"... get rid of "continue;"-stuff by moving stuff inside process...
        // classify vertices visibility with respect to the planes of the view frustum
        // we're in clip-coords (NDC), so just check if outside [-1, 1] x ...
        // Actually we're in clip-coords and it's not the same as NDC. We're only in NDC after the division by w.
        // We should do the clipping in clip-coords though. See
        // http://www.songho.ca/opengl/gl_projectionmatrix.html for more details.
        // However, when comparing against w_c below, we might run into the trouble of the sign again in the
        // affine case.

        // 'w' is always positive, as it is -z_camspace, and all z_camspace are negative.
        unsigned char visibility_bits[3];
        for (unsigned char k = 0; k < 3; k++)
        {
            visibility_bits[k] = 0;
            const float x_cc = clipspace_vertices[tri_indices[k]].position[0];
            const float y_cc = clipspace_vertices[tri_indices[k]].position[1];
            const float z_cc = clipspace_vertices[tri_indices[k]].position[2];
            const float w_cc = clipspace_vertices[tri_indices[k]].position[3];
            if (x_cc < -w_cc)            // true if outside of view frustum. False if on or inside the plane.
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
        // all vertices are visible - pass the whole triangle to the rasterizer. = All bits of all 3 triangles
        // are 0.
        if ((visibility_bits[0] | visibility_bits[1] | visibility_bits[2]) == 0)
        {
            cpp17::optional<detail::TriangleToRasterize> t = detail::process_prospective_tri(
                clipspace_vertices[tri_indices[0]], clipspace_vertices[tri_indices[1]],
                clipspace_vertices[tri_indices[2]], viewport_width, viewport_height, enable_backface_culling);
            if (t)
            {
                triangles_to_raster.push_back(*t);
            }
            continue;
        }
        // at this moment the triangle is known to be intersecting one of the view frustum's planes
        std::vector<detail::Vertex<float>> vertices;
        vertices.push_back(clipspace_vertices[tri_indices[0]]);
        vertices.push_back(clipspace_vertices[tri_indices[1]]);
        vertices.push_back(clipspace_vertices[tri_indices[2]]);
        // split the triangle if it intersects the near plane:
        if (enable_near_clipping)
        {
            vertices = detail::clip_polygon_to_plane_in_4d(
                vertices, glm::tvec4<float>(0.0f, 0.0f, -1.0f, -1.0f)); // "Normal" (or "4D hyperplane") of
                                                                        // the near-plane. I tested it and it
                                                                        // works like this but I'm a little
                                                                        // bit unsure because Songho says the
                                                                        // normal of the near-plane is
                                                                        // (0,0,-1,1) (maybe I have to switch
                                                                        // around the < 0 checks in the
                                                                        // function?)
        }

        // triangulation of the polygon formed of vertices array
        if (vertices.size() >= 3)
        {
            for (unsigned char k = 0; k < vertices.size() - 2; k++)
            {
                cpp17::optional<detail::TriangleToRasterize> t =
                    detail::process_prospective_tri(vertices[0], vertices[1 + k], vertices[2 + k],
                                                    viewport_width, viewport_height, enable_backface_culling);
                if (t)
                {
                    triangles_to_raster.push_back(*t);
                }
            }
        }
    }

    // Fragment/pixel shader: Colour the pixel values
    for (const auto& tri : triangles_to_raster)
    {
        detail::raster_triangle(tri, colorbuffer, depthbuffer, texture, enable_far_clipping);
    }
    return std::make_pair(colorbuffer, depthbuffer);
};

} /* namespace render */
} /* namespace eos */

#endif /* RENDER_HPP_ */
