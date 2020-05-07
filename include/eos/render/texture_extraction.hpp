/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/texture_extraction.hpp
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

#ifndef EOS_TEXTURE_EXTRACTION_HPP
#define EOS_TEXTURE_EXTRACTION_HPP

#include "eos/core/Image.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/render/detail/texture_extraction_detail.hpp"

// The following four includes are for v2::extract_texture(...):
#include "eos/render/transforms.hpp"
#include "eos/render/Rasterizer.hpp"
#include "eos/render/FragmentShader.hpp"
#include "eos/fitting/closest_edge_fitting.hpp" // for ray_triangle_intersect(). Move to eos/render/raycasting.hpp?

#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include <cassert>
#include <vector>
#include <cstddef>

namespace eos {
namespace render {

/* New texture extraction, will replace above one at some point: */
namespace v2 {

/**
 * @brief Extracts the texture of the face from the given image and stores it as isomap (a rectangular texture map).
 *
 * New texture extraction, will replace above one at some point.
 * Copy the documentation from above extract_texture function, once we replace it.
 *
 * Note/Todo: Add an overload that takes a vector of bool / visible vertices, for the case when we already computed the visibility? (e.g. for edge-fitting)
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] view_model_matrix Todo.
 * @param[in] projection_matrix Todo.
 * @param[in] viewport Not needed at the moment. Might be, if we change clip_to_screen_space() to take a viewport.
 * @param[in] image The image to extract the texture from. Todo: Does it have to be 8UC3 or something, or does it not matter?
 * @param[in] compute_view_angle Unused at the moment.
 * @param[in] isomap_resolution The resolution of the generated isomap. Defaults to 512x512.
 * @return The extracted texture as isomap (texture map).
 */
inline eos::core::Image4u
extract_texture(const core::Mesh& mesh, glm::mat4x4 view_model_matrix, glm::mat4x4 projection_matrix,
                glm::vec4 /*viewport, not needed at the moment */, const eos::core::Image4u& image,
                bool /* compute_view_angle, unused atm */, int isomap_resolution = 512)
{
    // Assert that either there are texture coordinates given for each vertex (in which case the texture map
    // doesn't contain any seams), or that a separate list of texture triangle indices is given (i.e. mesh.tti
    // is not empty):
    assert(mesh.vertices.size() == mesh.texcoords.size() || !mesh.tti.empty());
    // Sanity check on the texture triangle indices: They should be either empty or equal to tvi.size():
    assert(mesh.tti.empty() || mesh.tti.size() == mesh.tvi.size());

    using detail::divide_by_w;
    using glm::vec2;
    using glm::vec3;
    using glm::vec4;
    using std::vector;
    // actually we only need a rasteriser for this!
    Rasterizer<ExtractionFragmentShader> extraction_rasterizer(isomap_resolution, isomap_resolution);
    Texture image_to_extract_from_as_tex = create_mipmapped_texture(image, 1);
    extraction_rasterizer.enable_depth_test = false;
    extraction_rasterizer.extracting_tex = true;

    vector<bool> visibility_ray;
    vector<vec4> rotated_vertices;
    // In perspective case... does the perspective projection matrix not change visibility? Do we not need to
    // apply it?
    // (If so, then we can change the two input matrices to this function to one (mvp_matrix)).
    // Note 2019: I think it does and we need to take care of it, by changing the dot product, like we would
    // change it for perspective contour fitting.
    std::for_each(std::begin(mesh.vertices), std::end(mesh.vertices),
                  [&rotated_vertices, &view_model_matrix](auto&& v) {
                      rotated_vertices.push_back(view_model_matrix * glm::vec4(v.x(), v.y(), v.z(), 1.0));
                  });
    // This code is duplicated from the edge-fitting. I think I can put this into a function in the library.
    for (const auto& vertex : rotated_vertices)
    {
        bool visible = true;
        // For every tri of the rotated mesh:
        for (auto&& tri : mesh.tvi)
        {
            auto& v0 = rotated_vertices[tri[0]]; // const?
            auto& v1 = rotated_vertices[tri[1]];
            auto& v2 = rotated_vertices[tri[2]];

            const vec3 ray_origin(vertex);
            const vec3 ray_direction(0.0f, 0.0f, 1.0f); // we shoot the ray from the vertex towards the camera
            const auto intersect =
                ray_triangle_intersect(ray_origin, ray_direction, vec3(v0), vec3(v1), vec3(v2), false);
            // first is bool intersect, second is the distance t
            if (intersect.first == true)
            {
                // We've hit a triangle. Ray hit its own triangle. If it's behind the ray origin, ignore the
                // intersection:
                // Check if in front or behind?
                if (intersect.second.value() <= 1e-4)
                {
                    continue; // the intersection is behind the vertex, we don't care about it
                }
                // Otherwise, we've hit a genuine triangle, and the vertex is not visible:
                visible = false;
                break;
            }
        }
        visibility_ray.push_back(visible);
    }

    vector<vec4> wnd_coords; // will contain [x_wnd, y_wnd, z_ndc, 1/w_clip]
    for (auto&& vtx : mesh.vertices)
    {
        auto clip_coords = projection_matrix * view_model_matrix * vec4(vtx.x(), vtx.y(), vtx.z(), 1.0f);
        clip_coords = divide_by_w(clip_coords);
        const vec2 screen_coords = clip_to_screen_space(clip_coords.x, clip_coords.y, image.width(), image.height());
        clip_coords.x = screen_coords.x;
        clip_coords.y = screen_coords.y;
        wnd_coords.push_back(clip_coords);
    }

    // Go on with extracting: This only needs the rasteriser/FS, not the whole Renderer.
    const int tex_width = isomap_resolution;
    const int tex_height =
        isomap_resolution; // keeping this in case we need non-square texture maps at some point

    // Use Mesh::tti as texture triangle indices if present, tvi otherwise:
    const auto& mesh_tti = mesh.tti.empty() ? mesh.tvi : mesh.tti;

    for (std::size_t triangle_index = 0; triangle_index < mesh.tvi.size(); ++triangle_index)
    {
        // Select the three indices for the current triangle:
        const auto& tvi = mesh.tvi[triangle_index];
        const auto& tti = mesh_tti[triangle_index];

        // Check if all three vertices of the current triangle are visible, and use the triangle if so:
        if (visibility_ray[tvi[0]] && visibility_ray[tvi[1]] &&
            visibility_ray[tvi[2]]) // can also try using ||, but...
        {
            // The model's texcoords become the locations to extract to in the framebuffer (which is the
            // texture map we're extracting to). The wnd_coords are the coordinates we're extracting from (the
            // original image), which from the perspective of the rasteriser, is the texture map, and thus
            // from the rasteriser's perspective they're the texture coords.
            //
            // (Note: A test with a rendered & re-extracted texture showed that we're off by a pixel or more,
            //  definitely need to correct this. Probably here. It looks like it is 1-2 pixels off. Definitely
            //  a bit more than 1.)
            detail::Vertex<double> pa{
                vec4(mesh.texcoords[tti[0]][0] * tex_width,
					 mesh.texcoords[tti[0]][1] * tex_height,
                     wnd_coords[tvi[0]].z, // z_ndc
					 wnd_coords[tvi[0]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[0]].x / image.width(),
                    wnd_coords[tvi[0]].y / image.height() // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            detail::Vertex<double> pb{
                vec4(mesh.texcoords[tti[1]][0] * tex_width,
				mesh.texcoords[tti[1]][1] * tex_height,
                wnd_coords[tvi[1]].z, // z_ndc
				wnd_coords[tvi[1]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[1]].x / image.width(),
                    wnd_coords[tvi[1]].y / image.height() // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            detail::Vertex<double> pc{
                vec4(mesh.texcoords[tti[2]][0] * tex_width,
				mesh.texcoords[tti[2]][1] * tex_height,
                wnd_coords[tvi[2]].z, // z_ndc 
				wnd_coords[tvi[2]].w), // 1/w_clip
                vec3(), // empty
                vec2(
                    wnd_coords[tvi[2]].x / image.width(),
                    wnd_coords[tvi[2]].y / image.height() // (maybe '1 - wndcoords...'?) wndcoords of the projected/rendered model triangle (in the input img). Normalised to 0,1.
					)};
            // The wnd_coords (now p[a|b|c].texcoords) can actually be outside the image, if the head is
            // outside the image. Just skip the whole triangle if that is the case:
            if (pa.texcoords.x < 0 || pa.texcoords.x > 1 || pa.texcoords.y < 0 || pa.texcoords.y > 1 ||
                pb.texcoords.x < 0 || pb.texcoords.x > 1 || pb.texcoords.y < 0 || pb.texcoords.y > 1 ||
                pc.texcoords.x < 0 || pc.texcoords.x > 1 || pc.texcoords.y < 0 || pc.texcoords.y > 1)
            {
                continue;
            }
            extraction_rasterizer.raster_triangle(pa, pb, pc, image_to_extract_from_as_tex);
        }
    }

    return extraction_rasterizer.colorbuffer;
};

} /* namespace v2 */

} /* namespace render */
} /* namespace eos */

#endif /* EOS_TEXTURE_EXTRACTION_HPP */
