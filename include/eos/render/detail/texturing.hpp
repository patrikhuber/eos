/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/texturing.hpp
 *
 * Copyright 2014-2017, 2023 Patrik Huber
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

#ifndef EOS_TEXTURING_DETAIL_HPP
#define EOS_TEXTURING_DETAIL_HPP

#include "eos/render/Texture.hpp"

#include "Eigen/Core"

#include <algorithm>
#include <cmath>

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
namespace render {
namespace detail {

// used only in tex2D_linear_mipmap_linear
// template? or we could use std::clamp(), with C++17.
inline float clamp(float x, float a, float b)
{
    return std::max(std::min(x, b), a);
};

template <typename T>
inline Eigen::Vector2<T> texcoord_wrap(const Eigen::Vector2<T>& texcoords)
{
    return Eigen::Vector2<T>(texcoords[0] - (int)texcoords[0], texcoords[1] - (int)texcoords[1]);
};

// forward declarations:
Eigen::Vector3f tex2d_linear_mipmap_linear(const Eigen::Vector2f& texcoords, const Texture& texture,
                                           float dudx, float dudy, float dvdx, float dvdy);
template <typename T>
inline Eigen::Vector3<T> tex2d_linear(const Eigen::Vector2<T>& imageTexCoord, unsigned char mipmap_index,
                                      const Texture& texture);

inline Eigen::Vector3f tex2d(const Eigen::Vector2f& texcoords, const Texture& texture, float dudx, float dudy, float dvdx,
                       float dvdy)
{
    return (1.0f / 255.0f) * tex2d_linear_mipmap_linear(texcoords, texture, dudx, dudy, dvdx, dvdy);
};

template <typename T, glm::precision P = glm::defaultp>
glm::tvec3<T, P> tex2d(const glm::tvec2<T, P>& texcoords, const Texture& texture, float dudx, float dudy,
                       float dvdx, float dvdy)
{
    Eigen::Vector3f ret = (1.0f / 255.0f) * tex2d_linear_mipmap_linear(Eigen::Vector2f(texcoords[0], texcoords[1]),
                                                                 texture, dudx, dudy, dvdx, dvdy);
    return glm::tvec3<T, P>(ret[0], ret[1], ret[2]);
};

inline Eigen::Vector3f tex2d_linear_mipmap_linear(const Eigen::Vector2f& texcoords, const Texture& texture,
                                                  float dudx, float dudy, float dvdx, float dvdy)
{
    using Eigen::Vector2f;
    const float px = std::sqrt(std::pow(dudx, 2) + std::pow(dvdx, 2));
    const float py = std::sqrt(std::pow(dudy, 2) + std::pow(dvdy, 2));
    constexpr double ln2 = 0.69314718056;
    const float lambda = std::log(std::max(px, py)) / static_cast<float>(ln2);
    const unsigned char mipmapIndex1 =
        detail::clamp((int)lambda, 0.0f, std::max(texture.widthLog, texture.heightLog) - 1);
    const unsigned char mipmapIndex2 = mipmapIndex1 + 1;

    const Vector2f imageTexCoord = detail::texcoord_wrap(texcoords);
    Vector2f imageTexCoord1 = imageTexCoord;
    imageTexCoord1[0] *= texture.mipmaps[mipmapIndex1].width();
    imageTexCoord1[1] *= texture.mipmaps[mipmapIndex1].height();
    Vector2f imageTexCoord2 = imageTexCoord;
    imageTexCoord2[0] *= texture.mipmaps[mipmapIndex2].width();
    imageTexCoord2[1] *= texture.mipmaps[mipmapIndex2].height();

    Eigen::Vector3f color, color1, color2;
    color1 = tex2d_linear(imageTexCoord1, mipmapIndex1, texture);
    color2 = tex2d_linear(imageTexCoord2, mipmapIndex2, texture);
    float lambdaFrac = std::max(lambda, 0.0f);
    lambdaFrac = lambdaFrac - (int)lambdaFrac;
    color = (1.0f - lambdaFrac) * color1 + lambdaFrac * color2;

    return color;
};

template <typename T>
inline Eigen::Vector3<T> tex2d_linear(const Eigen::Vector2<T>& imageTexCoord, unsigned char mipmap_index,
                                      const Texture& texture)
{
    const int x = (int)imageTexCoord[0];
    const int y = (int)imageTexCoord[1];
    const float alpha = imageTexCoord[0] - x;
    const float beta = imageTexCoord[1] - y;
    const float oneMinusAlpha = 1.0f - alpha;
    const float oneMinusBeta = 1.0f - beta;
    const float a = oneMinusAlpha * oneMinusBeta;
    const float b = alpha * oneMinusBeta;
    const float c = oneMinusAlpha * beta;
    const float d = alpha * beta;
    Eigen::Vector3<T> color;

    // int pixelIndex;
    // pixelIndex = getPixelIndex_wrap(x, y, texture->mipmaps[mipmapIndex].cols,
    // texture->mipmaps[mipmapIndex].rows);
    int pixelIndexCol = x;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].width())
    {
        pixelIndexCol = 0;
    }
    int pixelIndexRow = y;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].height())
    {
        pixelIndexRow = 0;
    }
    // std::cout << texture.mipmaps[mipmapIndex].cols << " " << texture.mipmaps[mipmapIndex].rows << " " <<
    // texture.mipmaps[mipmapIndex].channels() << std::endl;  cv::imwrite("mm.png",
    // texture.mipmaps[mipmapIndex]);
    color[0] = a * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[0];
    color[1] = a * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[1];
    color[2] = a * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x + 1, y, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x + 1;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].width())
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].height())
    {
        pixelIndexRow = 0;
    }
    color[0] += b * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[0];
    color[1] += b * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[1];
    color[2] += b * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x, y + 1, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].width())
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y + 1;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].height())
    {
        pixelIndexRow = 0;
    }
    color[0] += c * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[0];
    color[1] += c * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[1];
    color[2] += c * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x + 1, y + 1, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x + 1;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].width())
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y + 1;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].height())
    {
        pixelIndexRow = 0;
    }
    color[0] += d * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[0];
    color[1] += d * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[1];
    color[2] += d * texture.mipmaps[mipmap_index](pixelIndexRow, pixelIndexCol)[2];

    return color;
};

} /* namespace detail */
} /* namespace render */
} /* namespace eos */

#endif /* EOS_TEXTURING_DETAIL_HPP */
