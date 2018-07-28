/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/image/PixelTraits.hpp
 *
 * Copyright 2018 Patrik Huber
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

#ifndef EOS_IMAGE_PIXELTRAITS_HPP
#define EOS_IMAGE_PIXELTRAITS_HPP

#include "eos/core/image/Pixel.hpp"

#include <cstdint>

namespace eos {
namespace core {

/**
 * @brief Todo.
 */
template <typename ElementType>
struct PixelTraits
{
    static constexpr std::uint16_t num_bytes = sizeof(ElementType);

	static constexpr ElementType zero_element = ElementType{0};
};

/**
 * @brief Todo.
 */
template <typename ElementType, int NumChannels>
struct PixelTraits<Pixel<ElementType, NumChannels>>
{
    static constexpr std::uint16_t num_bytes = sizeof(Pixel<ElementType, NumChannels>);

    static constexpr Pixel<ElementType, NumChannels> zero_element =
        Pixel<ElementType, NumChannels>(std::array<ElementType, NumChannels>());
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_IMAGE_PIXELTRAITS_HPP */
