/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/image/utils.hpp
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

#ifndef EOS_IMAGE_UTILS_HPP
#define EOS_IMAGE_UTILS_HPP

#include "eos/core/image/Pixel.hpp"

#include <cstdint>

namespace eos {
namespace core {
namespace image {

/**
 * @brief Creates an image of given type, height and width, with all zeros.
 */
template <typename PixelType>
Image<PixelType> zeros(int height, int width) noexcept
{
    Image<PixelType> image(height, width);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            image(y, x) = PixelTraits<PixelType>::zero_element;
        }
    }
    return image;
};

/**
 * @brief Creates an image of given type, height and width, with a constant value for all pixels.
 */
template <typename PixelType>
Image<PixelType> constant(int height, int width, PixelType value) noexcept
{
    Image<PixelType> image(height, width);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            image(y, x) = value;
        }
    }
    return image;
};

} // namespace image
} // namespace core
} // namespace eos

#endif /* EOS_IMAGE_UTILS_HPP */
