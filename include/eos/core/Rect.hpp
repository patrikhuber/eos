/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Rect.hpp
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

#ifndef EOS_RECT_HPP
#define EOS_RECT_HPP

namespace eos {
namespace core {

/**
 * @brief A simple type representing a rectangle.
 */
template <typename T>
struct Rect
{
    T x, y; ///< Top-left corner x and y position
    T width, height;
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_RECT_HPP */
