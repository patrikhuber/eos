/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/render.hpp
 *
 * Copyright 2014-2020, 2023 Patrik Huber
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

#ifndef EOS_RENDER_HPP
#define EOS_RENDER_HPP

#include "eos/core/Image.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/render/SoftwareRenderer.hpp"
#include "eos/render/VertexShader.hpp"
#include "eos/render/FragmentShader.hpp"

#include "Eigen/Core"

namespace eos {
namespace render {

/**
 * Convenience function that renders the given mesh onto a 2D image using \c SoftwareRenderer. Conforms to
 * OpenGL conventions.
 *
 * Renders using per-vertex colouring, without texturing.
 *
 * @param[in] mesh A 3D mesh.
 * @param[in] model_view_matrix A 4x4 OpenGL model-view matrix.
 * @param[in] projection_matrix A 4x4 orthographic or perspective OpenGL projection matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] enable_backface_culling Whether the renderer should perform backface culling. If true, only draw triangles with vertices ordered CCW in screen-space.
 * @param[in] enable_near_clipping Whether vertices should be clipped against the near plane.
 * @param[in] enable_far_clipping Whether vertices should be clipped against the far plane.
 * @return Framebuffer (colourbuffer) with the rendered image.
 */
core::Image4u render(const core::Mesh& mesh, const Eigen::Matrix4f& model_view_matrix,
                     const Eigen::Matrix4f& projection_matrix, int viewport_width, int viewport_height,
                     bool enable_backface_culling = false, bool enable_near_clipping = true,
                     bool enable_far_clipping = true)
{
    SoftwareRenderer<VertexShader, VertexColoringFragmentShader> software_renderer(viewport_width,
                                                                                   viewport_height);
    software_renderer.enable_backface_culling = enable_backface_culling;
    software_renderer.enable_near_clipping = enable_near_clipping;
    software_renderer.rasterizer.enable_far_clipping = enable_far_clipping;

    return software_renderer.render(mesh, model_view_matrix, projection_matrix);
};

/**
 * Convenience function that renders the given mesh onto a 2D image using \c SoftwareRenderer. Conforms to
 * OpenGL conventions.
 *
 * Performs texturing using the given \p texture.
 *
 * @param[in] mesh A 3D mesh.
 * @param[in] model_view_matrix A 4x4 OpenGL model-view matrix.
 * @param[in] projection_matrix A 4x4 orthographic or perspective OpenGL projection matrix.
 * @param[in] viewport_width Screen width.
 * @param[in] viewport_height Screen height.
 * @param[in] texture A texture map to texture the model.
 * @param[in] enable_backface_culling Whether the renderer should perform backface culling. If true, only draw
 * triangles with vertices ordered CCW in screen-space.
 * @param[in] enable_near_clipping Whether vertices should be clipped against the near plane.
 * @param[in] enable_far_clipping Whether vertices should be clipped against the far plane.
 * @return Framebuffer (colourbuffer) with the rendered image.
 */
core::Image4u render(const core::Mesh& mesh, const Eigen::Matrix4f& model_view_matrix,
                     const Eigen::Matrix4f& projection_matrix, int viewport_width, int viewport_height,
                     Texture texture, bool enable_backface_culling = false, bool enable_near_clipping = true,
                     bool enable_far_clipping = true)
{
    SoftwareRenderer<VertexShader, TexturingFragmentShader> software_renderer(viewport_width,
                                                                              viewport_height);
    software_renderer.enable_backface_culling = enable_backface_culling;
    software_renderer.enable_near_clipping = enable_near_clipping;
    software_renderer.rasterizer.enable_far_clipping = enable_far_clipping;

    return software_renderer.render(mesh, model_view_matrix, projection_matrix, texture);
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_RENDER_HPP */
