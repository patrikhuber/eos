/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/ProjectionType.hpp
 *
 * Copyright 2019, 2020 Patrik Huber
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

#ifndef EOS_RENDER_PROJECTION_TYPE_HPP
#define EOS_RENDER_PROJECTION_TYPE_HPP

namespace eos {
namespace render {

/**
 * Specifies whether orthographic or perspective projection shall be used.
 */
enum class ProjectionType { Orthographic, Perspective };

} // namespace render
} // namespace eos

#endif /* EOS_RENDER_PROJECTION_TYPE_HPP */
