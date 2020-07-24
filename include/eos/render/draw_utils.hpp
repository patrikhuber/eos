/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/draw_utils.hpp
 *
 * Copyright 2017-2019 Patrik Huber
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

#ifndef EOS_RENDER_DRAW_UTILS_HPP
#define EOS_RENDER_DRAW_UTILS_HPP

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/render_detail_utils.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"

namespace eos {
namespace render {

/**
 * Draws a line using the Bresenham algorithm.
 *
 * From: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
 * I also tried this: https://www.thecrazyprogrammer.com/2017/01/bresenhams-line-drawing-algorithm-c-c.html
 * which looks awesome, but it drew weird lines.
 *
 * @param[in] image An image to draw into.
 * @param[in] x0 X coordinate of the start point.
 * @param[in] y0 Y coordinate of the start point.
 * @param[in] x1 X coordinate of the start point.
 * @param[in] y1 Y coordinate of the end point.
 * @param[in] color RGB colour of the line to be drawn.
 */

inline void draw_line(core::Image3u& image, float x0, float y0, float x1, float y1, glm::vec3 color)
{
    auto plot_line_low = [&image, &color](float x0, float y0, float x1, float y1) {
        float dx = x1 - x0;
        float dy = y1 - y0;
        int yi = 1;
        if (dy < 0)
        {
            yi = -1;
            dy = -dy;
        }

        float D = 2 * dy - dx;
        float y = y0;

        for (int x = x0; x <= x1; ++x) // for x from x0 to x1
        {
            image(y, x) = {color[0], color[1], color[2]}; // plot(x, y);
            if (D > 0)
            {
                y = y + yi;
                D = D - 2 * dx;
            }
            D = D + 2 * dy;
        }
    };

    auto plot_line_high = [&image, &color](float x0, float y0, float x1, float y1) {
        float dx = x1 - x0;
        float dy = y1 - y0;
        int xi = 1;
        if (dx < 0)
        {
            xi = -1;
            dx = -dx;
        }

        float D = 2 * dx - dy;
        float x = x0;

        for (int y = y0; y <= y1; ++y) // for y from y0 to y1
        {
            image(y, x) = {color[0], color[1], color[2]}; // plot(x, y);
            if (D > 0)
            {
                x = x + xi;
                D = D - 2 * dy;
            }
            D = D + 2 * dx;
        }
    };

    if (abs(y1 - y0) < abs(x1 - x0))
    {
        if (x0 > x1)
        {
            plot_line_low(x1, y1, x0, y0);
        } else
        {
            plot_line_low(x0, y0, x1, y1);
        }
    } else
    {
        if (y0 > y1)
        {
            plot_line_high(x1, y1, x0, y0);
        } else
        {
            plot_line_high(x0, y0, x1, y1);
        }
    }
};

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] color Colour of the mesh to be drawn, in RGB.
 */
inline void draw_wireframe(core::Image3u& image, const core::Mesh& mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport, glm::vec3 color = glm::vec3(0, 255, 0))
{
    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);
        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
            draw_line(image, p1.x, p1.y, p2.x, p2.y, color);
            draw_line(image, p2.x, p2.y, p3.x, p3.y, color);
            draw_line(image, p3.x, p3.y, p1.x, p1.y, color);
        }
    }
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_RENDER_DRAW_UTILS_HPP */
