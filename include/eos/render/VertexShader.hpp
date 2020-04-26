/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/VertexShader.hpp
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

#ifndef EOS_VERTEX_SHADER_HPP
#define EOS_VERTEX_SHADER_HPP

#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"

namespace eos {
namespace render {

/**
 * @brief A simple vertex shader that projects the vertex and returns the vertex in clip-space coordinates.
 */
class VertexShader
{
public:
    /**
     * @brief Projects the given vertex into clip-space and returns it.
     *
     * @param[in] vertex The vertex to project.
     * @param[in] model_view_matrix The model-view matrix.
     * @param[in] projection_matrix The projection matrix.
     * @tparam VT Vertex type.
     * @tparam VP Vertex precision.
     * @tparam MT Matrix type.
     * @tparam MP Matrix precision.
     * @return Vertex projected to clip space.
     */
    template <typename VT, glm::precision VP, typename MT, glm::precision MP = glm::defaultp>
    glm::tvec4<MT, MP> operator()(const glm::tvec4<VT, VP>& vertex,
                                  const glm::tmat4x4<MT, MP>& model_view_matrix,
                                  const glm::tmat4x4<MT, MP>& projection_matrix)
    {
        return projection_matrix * model_view_matrix * vertex;
    };
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_VERTEX_SHADER_HPP */
