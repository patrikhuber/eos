/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/Texture.hpp
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

#ifndef EOS_TEXTURE_HPP
#define EOS_TEXTURE_HPP

#include "eos/core/Image.hpp"
#include "eos/core/image/resize.hpp"

#include <algorithm>
#include <vector>

namespace eos {
namespace render {

// TODO: Should go to detail:: namespace, or texturing/utils or whatever.
inline unsigned int get_max_possible_mipmaps_num(unsigned int width, unsigned int height)
{
    unsigned int mipmapsNum = 1;
    unsigned int size = std::max(width, height);

    if (size == 1)
        return 1;

    do
    {
        size >>= 1;
        mipmapsNum++;
    } while (size != 1);

    return mipmapsNum;
};

inline bool is_power_of_two(int x) { return !(x & (x - 1)); };

/**
 * @brief Represents a texture for rendering.
 *
 * Represents a texture and mipmap levels for use in the renderer.
 * Todo: This whole class needs a major overhaul and documentation.
 */
class Texture
{
public:
    std::vector<eos::core::Image4u> mipmaps; // make Texture a friend class of renderer, then move this to private?
    unsigned char widthLog, heightLog; // log2 of width and height of the base mip-level

    // private:
    // std::string filename;
    unsigned int mipmaps_num;
};

// throws: ocv exc,  runtime_ex
inline Texture create_mipmapped_texture(const eos::core::Image4u& image, unsigned int mipmapsNum = 0)
{
    Texture texture;

    texture.mipmaps_num =
        (mipmapsNum == 0 ? get_max_possible_mipmaps_num(image.width(), image.height()) : mipmapsNum);
    /*if (mipmapsNum == 0)
    {
    uchar mmn = render::utils::MatrixUtils::getMaxPossibleMipmapsNum(image.cols, image.rows);
    this->mipmapsNum = mmn;
    } else
    {
    this->mipmapsNum = mipmapsNum;
    }*/

    if (texture.mipmaps_num > 1)
    {
        if (!is_power_of_two(image.width()) || !is_power_of_two(image.height()))
        {
            throw std::runtime_error("Error: Couldn't generate mipmaps, width or height not power of two.");
        }
    }

    int currWidth = image.width();
    int currHeight = image.height();
    std::vector<eos::core::Image4u> mipmaps;
    for (unsigned int i = 0; i < texture.mipmaps_num; i++)
    {
        if (i == 0)
        {
            mipmaps.push_back(image);
        } else
        {
            const eos::core::Image4u currMipMap =
                eos::core::image::resize(mipmaps[i - 1], currWidth, currHeight);
            mipmaps.push_back(currMipMap);
        }

        if (currWidth > 1)
            currWidth >>= 1;
        if (currHeight > 1)
            currHeight >>= 1;
    }
    texture.mipmaps = mipmaps;
	constexpr double ln2 = 0.69314718056;
    texture.widthLog = (unsigned char)(std::log(mipmaps[0].width()) / ln2 +
                               0.0001f); // std::epsilon or something? or why 0.0001f here?
    texture.heightLog = (unsigned char)(
        std::log(mipmaps[0].height()) / ln2 +
        0.0001f); // Changed std::logf to std::log because it doesnt compile in linux (gcc 4.8). CHECK THAT
    return texture;
};

} /* namespace render */
} /* namespace eos */

#endif /* EOS_TEXTURE_HPP */
