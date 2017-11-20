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
#include "eos/render/Rasterizer.hpp"
#include "eos/render/detail/Vertex.hpp"
#include "eos/render/detail/render_detail.hpp"
#include "eos/render/utils.hpp" // for Texture, potentially others

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec4.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <array>
#include <memory>

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
std::vector<detail::Vertex<T, P>>
clip_polygon_to_plane_in_4d(const std::vector<detail::Vertex<T, P>>& vertices,
                            const glm::tvec4<T, P>& plane_normal);

/**
 * @brief X.
 *
 * Can this go into the SoftwareRenderer class or something? No, I think FragShader needs it? Where to put it?
 */
template <typename T, glm::precision P = glm::defaultp>
using Triangle = std::array<detail::Vertex<T, P>, 3>;

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
    {
        rasterizer = std::make_unique<Rasterizer<FragmentShaderType>>(viewport_width, viewport_height);
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
    cv::Mat render(const core::Mesh& mesh, const glm::tmat4x4<T, P>& model_view_matrix,
                   const glm::tmat4x4<T, P>& projection_matrix,
                   const boost::optional<Texture>& texture = boost::none)
    {
        assert(mesh.vertices.size() == mesh.colors.size() ||
               mesh.colors.empty()); // The number of vertices has to be equal for both shape and colour, or,
                                     // alternatively, it has to be a shape-only model.
        assert(mesh.vertices.size() == mesh.texcoords.size() ||
               mesh.texcoords.empty()); // same for the texcoords
        // Add another assert: If cv::Mat texture != empty (and/or texturing=true?), then we need texcoords?

        using cv::Mat;
        using detail::divide_by_w;
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
                if (rasterizer->enable_far_clipping && z_cc > w_cc) // far plane frustum clipping
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
                const glm::tvec2<T, P> v0_screen =
                    clip_to_screen_space(prospective_tri[0].x, prospective_tri[0].y,
                                         rasterizer->viewport_width, rasterizer->viewport_height);
                prospective_tri[0].x = v0_screen.x;
                prospective_tri[0].y = v0_screen.y;
                const glm::tvec2<T, P> v1_screen =
                    clip_to_screen_space(prospective_tri[1].x, prospective_tri[1].y,
                                         rasterizer->viewport_width, rasterizer->viewport_height);
                prospective_tri[1].x = v1_screen.x;
                prospective_tri[1].y = v1_screen.y;
                const glm::tvec2<T, P> v2_screen =
                    clip_to_screen_space(prospective_tri[2].x, prospective_tri[2].y,
                                         rasterizer->viewport_width, rasterizer->viewport_height);
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
                    glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y), rasterizer->viewport_width,
                    rasterizer->viewport_height);
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
                    Triangle<T, P>{detail::Vertex<T, P>{prospective_tri[0], mesh.colors[tri_indices[0]],
                                                        mesh.texcoords[tri_indices[0]]},
                                   detail::Vertex<T, P>{prospective_tri[1], mesh.colors[tri_indices[1]],
                                                        mesh.texcoords[tri_indices[1]]},
                                   detail::Vertex<T, P>{prospective_tri[2], mesh.colors[tri_indices[2]],
                                                        mesh.texcoords[tri_indices[2]]}});
                continue; // Triangle was either added or not added. Continue with next triangle.
            }
            // At this point, the triangle is known to be intersecting one of the view frustum's planes
            // Note: It seems that this is only w.r.t. the near-plane. If a triangle is partially outside the
            // tlbr viewport, it'll get rejected.
            // Well, 'z' of these triangles seems to be -1, so is that really the near plane?
            std::vector<detail::Vertex<T, P>> vertices;
            vertices.reserve(3);
            vertices.push_back(detail::Vertex<T, P>{clipspace_vertices[tri_indices[0]],
                                                    mesh.colors[tri_indices[0]],
                                                    mesh.texcoords[tri_indices[0]]});
            vertices.push_back(detail::Vertex<T, P>{clipspace_vertices[tri_indices[1]],
                                                    mesh.colors[tri_indices[1]],
                                                    mesh.texcoords[tri_indices[1]]});
            vertices.push_back(detail::Vertex<T, P>{clipspace_vertices[tri_indices[2]],
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

                    const glm::tvec2<T, P> v0_screen =
                        clip_to_screen_space(prospective_tri[0].x, prospective_tri[0].y,
                                             rasterizer->viewport_width, rasterizer->viewport_height);
                    prospective_tri[0].x = v0_screen.x;
                    prospective_tri[0].y = v0_screen.y;
                    const glm::tvec2<T, P> v1_screen =
                        clip_to_screen_space(prospective_tri[1].x, prospective_tri[1].y,
                                             rasterizer->viewport_width, rasterizer->viewport_height);
                    prospective_tri[1].x = v1_screen.x;
                    prospective_tri[1].y = v1_screen.y;
                    const glm::tvec2<T, P> v2_screen =
                        clip_to_screen_space(prospective_tri[2].x, prospective_tri[2].y,
                                             rasterizer->viewport_width, rasterizer->viewport_height);
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
                        glm::tvec2<T, P>(prospective_tri[2].x, prospective_tri[2].y),
                        rasterizer->viewport_width, rasterizer->viewport_height);
                    const auto min_x = boundingBox.x;
                    const auto max_x = boundingBox.x + boundingBox.width;
                    const auto min_y = boundingBox.y;
                    const auto max_y = boundingBox.y + boundingBox.height;
                    if (max_x <= min_x || max_y <= min_y)
                    {
                        continue;
                    }

                    // If we're here, the triangle is CCW in screen space and the bbox is inside the viewport!
                    triangles_to_raster.push_back(Triangle<T, P>{
                        detail::Vertex<T, P>{prospective_tri[0], vertices[0].color, vertices[0].texcoords},
                        detail::Vertex<T, P>{prospective_tri[1], vertices[1 + k].color,
                                             vertices[1 + k].texcoords},
                        detail::Vertex<T, P>{prospective_tri[2], vertices[2 + k].color,
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
            rasterizer->raster_triangle(tri[0], tri[1], tri[2], texture);
        }
        return rasterizer->colorbuffer;
    };

public: // Todo: these should go private in the final implementation
    boost::optional<Texture> texture = boost::none;
    bool enable_backface_culling = false;
    bool enable_near_clipping = true;

    std::unique_ptr<Rasterizer<FragmentShaderType>> rasterizer; // Rasterizer is not default-constructible
private:
    VertexShaderType vertex_shader;
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
std::vector<detail::Vertex<T, P>>
clip_polygon_to_plane_in_4d(const std::vector<detail::Vertex<T, P>>& vertices,
                            const glm::tvec4<T, P>& plane_normal)
{
    std::vector<detail::Vertex<T, P>> clipped_vertices;

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
                clipped_vertices.push_back(detail::Vertex<T, P>{position, color, texcoords});
            } else if (fb < 0) // we use only the new vertex
            {
                clipped_vertices.push_back(detail::Vertex<T, P>{position, color, texcoords});
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
