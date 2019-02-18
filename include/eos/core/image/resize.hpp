/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/image/resize.hpp
 *
 * Copyright 2018-2019 Patrik Huber
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

#ifndef EOS_IMAGE_RESIZE_HPP
#define EOS_IMAGE_RESIZE_HPP

#include "eos/core/Image.hpp"

#include "Eigen/Dense"

#include <cstdint>
#include <cmath>

namespace eos {
namespace core {
namespace image {

/**
 * @brief Resize an image using area interpolation.
 *
 * This code was contributed by Philipp Kopp (@PhilippKopp). The main part of the code was copied from the
 * interpolation code in the texture extraction.
 * Todo/Question: Are there any preconditions on width and height? Do they have to be a factor of two from the
 * original image, or something like that?
 *
 * @param[in] image The image to resize.
 * @param[in] width Todo.
 * @param[in] height Todo.
 * @return The resized image.
 */
inline eos::core::Image3u resize(const eos::core::Image3u& image, int width, int height)
{
    using Eigen::Vector3i;
    using std::ceil;
    using std::floor;
    using std::round;
    using std::uint8_t;

    const float scaling_width = static_cast<float>(width) / image.width();
    const float scaling_height = static_cast<float>(height) / image.height();

    eos::core::Image3u resized_image(height, width);
    for (int w = 0; w < width; ++w)
    {
        for (int h = 0; h < height; ++h)
        {
            const float left = (static_cast<float>(w)) / scaling_width;
            const float right = (static_cast<float>(w) + 1.0f) / scaling_width;
            const float bottom = (static_cast<float>(h)) / scaling_height;
            const float top = (static_cast<float>(h) + 1.0f) / scaling_height;

            Vector3i color = Vector3i::Zero(); // std::uint8_t actually.
            int num_texels = 0;
            // loop over square in which quadrangle out of the four corners of pixel is
            for (int a = ceil(left); a <= floor(right); ++a)
            {
                for (int b = ceil(bottom); b <= floor(top); ++b)
                {
                    // check if texel is in image
                    if (a < image.width() && b < image.height())
                    {
                        num_texels++;
                        color += Vector3i(image(b, a)[0], image(b, a)[1], image(b, a)[2]);
                    }
                }
            }
            if (num_texels > 0)
                color = color / num_texels;
            else
            { // if no corresponding texel found, nearest neighbour interpolation
              // calculate corresponding position of dst_coord pixel center in image (src)
                const int source_x = round(static_cast<float>(w) / scaling_width);
                const int source_y = round(static_cast<float>(h) / scaling_height);

                if (source_y < image.height() && source_x < image.width())
                {
                    color = Vector3i(image(source_y, source_x)[0], image(source_y, source_x)[1],
                                     image(source_y, source_x)[2]);
                }
            }

            resized_image(h, w) = {static_cast<std::uint8_t>(round(color[0])),
                                   static_cast<std::uint8_t>(round(color[1])),
                                   static_cast<std::uint8_t>(round(color[2]))};
        }
    }

    return resized_image;
};

/**
 * @brief Resize an image using area interpolation.
 *
 * @see resize(const eos::core::Image3u& image, int width, int height).
 *
 * Note/Todo: This is an exact copy of the Image3u function, adjusted for Image4u. We should templatise and
 * avoid code duplication.
 *
 * @param[in] image The image to resize.
 * @param[in] width Todo.
 * @param[in] height Todo.
 * @return The resized image.
 */
inline eos::core::Image4u resize(const eos::core::Image4u& image, int width, int height)
{
    using Eigen::Vector4i;
    using std::ceil;
    using std::floor;
    using std::round;
    using std::uint8_t;

    const float scaling_width = static_cast<float>(width) / image.width();
    const float scaling_height = static_cast<float>(height) / image.height();

    eos::core::Image4u resized_image(height, width);
    for (int w = 0; w < width; ++w)
    {
        for (int h = 0; h < height; ++h)
        {
            const float left = (static_cast<float>(w)) / scaling_width;
            const float right = (static_cast<float>(w) + 1.0f) / scaling_width;
            const float bottom = (static_cast<float>(h)) / scaling_height;
            const float top = (static_cast<float>(h) + 1.0f) / scaling_height;

            Vector4i color = Vector4i::Zero(); // std::uint8_t actually.
            int num_texels = 0;
            // loop over square in which quadrangle out of the four corners of pixel is
            for (int a = ceil(left); a <= floor(right); ++a)
            {
                for (int b = ceil(bottom); b <= floor(top); ++b)
                {
                    // check if texel is in image
                    if (a < image.width() && b < image.height())
                    {
                        num_texels++;
                        color += Vector4i(image(b, a)[0], image(b, a)[1], image(b, a)[2], image(b, a)[3]);
                    }
                }
            }
            if (num_texels > 0)
                color = color / num_texels;
            else
            { // if no corresponding texel found, nearest neighbour interpolation
              // calculate corresponding position of dst_coord pixel center in image (src)
                const int source_x = round(static_cast<float>(w) / scaling_width);
                const int source_y = round(static_cast<float>(h) / scaling_height);

                if (source_y < image.height() && source_x < image.width())
                {
                    color = Vector4i(image(source_y, source_x)[0], image(source_y, source_x)[1],
                                     image(source_y, source_x)[2], image(source_y, source_x)[3]);
                }
            }

            resized_image(h, w) = {
                static_cast<std::uint8_t>(round(color[0])), static_cast<std::uint8_t>(round(color[1])),
                static_cast<std::uint8_t>(round(color[2])), static_cast<std::uint8_t>(round(color[3]))};
        }
    }

    return resized_image;
};

} // namespace image
} // namespace core
} // namespace eos

#endif /* EOS_IMAGE_RESIZE_HPP */
