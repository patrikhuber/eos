/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/detail/texturing.hpp
 *
 * Copyright 2014-2017 Patrik Huber
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

#ifndef TEXTURING_DETAIL_HPP_
#define TEXTURING_DETAIL_HPP_

//#include "eos/render/utils.hpp"
#include "eos/render/Texture.hpp"

#include "glm/glm.hpp"

#include "opencv2/core/core.hpp"

/**
 * Implementations of internal functions, not part of the
 * API we expose and not meant to be used by a user.
 */
namespace eos {
namespace render {
namespace detail {

// used only in tex2D_linear_mipmap_linear
// template?
inline float clamp(float x, float a, float b)
{
    return std::max(std::min(x, b), a);
};

inline glm::vec2 texcoord_wrap(const glm::vec2& texcoords)
{
    return glm::vec2(texcoords[0] - (int)texcoords[0], texcoords[1] - (int)texcoords[1]);
};

// forward decls
glm::vec3 tex2d_linear_mipmap_linear(const glm::vec2& texcoords, const Texture& texture, float dudx,
                                     float dudy, float dvdx, float dvdy);
glm::vec3 tex2d_linear(const glm::vec2& imageTexCoord, unsigned char mipmapIndex, const Texture& texture);

inline glm::vec3 tex2d(const glm::vec2& texcoords, const Texture& texture, float dudx, float dudy, float dvdx,
                       float dvdy)
{
    return (1.0f / 255.0f) * tex2d_linear_mipmap_linear(texcoords, texture, dudx, dudy, dvdx, dvdy);
};

template <typename T, glm::precision P = glm::defaultp>
glm::tvec3<T, P> tex2d(const glm::tvec2<T, P>& texcoords, const Texture& texture, float dudx, float dudy,
                       float dvdx, float dvdy)
{
    // Todo: Change everything to GLM.
    glm::vec3 ret = (1.0f / 255.0f) * tex2d_linear_mipmap_linear(glm::vec2(texcoords[0], texcoords[1]),
                                                                 texture, dudx, dudy, dvdx, dvdy);
    return glm::tvec3<T, P>(ret[0], ret[1], ret[2]);
};

inline glm::vec3 tex2d_linear_mipmap_linear(const glm::vec2& texcoords, const Texture& texture, float dudx,
                                            float dudy, float dvdx, float dvdy)
{
    using glm::vec2;
    const float px = std::sqrt(std::pow(dudx, 2) + std::pow(dvdx, 2));
    const float py = std::sqrt(std::pow(dudy, 2) + std::pow(dvdy, 2));
    const float lambda = std::log(std::max(px, py)) / CV_LOG2;
    const unsigned char mipmapIndex1 =
        detail::clamp((int)lambda, 0.0f, std::max(texture.widthLog, texture.heightLog) - 1);
    const unsigned char mipmapIndex2 = mipmapIndex1 + 1;

    const vec2 imageTexCoord = detail::texcoord_wrap(texcoords);
    vec2 imageTexCoord1 = imageTexCoord;
    imageTexCoord1[0] *= texture.mipmaps[mipmapIndex1].cols;
    imageTexCoord1[1] *= texture.mipmaps[mipmapIndex1].rows;
    vec2 imageTexCoord2 = imageTexCoord;
    imageTexCoord2[0] *= texture.mipmaps[mipmapIndex2].cols;
    imageTexCoord2[1] *= texture.mipmaps[mipmapIndex2].rows;

    glm::vec3 color, color1, color2;
    color1 = tex2d_linear(imageTexCoord1, mipmapIndex1, texture);
    color2 = tex2d_linear(imageTexCoord2, mipmapIndex2, texture);
    float lambdaFrac = std::max(lambda, 0.0f);
    lambdaFrac = lambdaFrac - (int)lambdaFrac;
    color = (1.0f - lambdaFrac) * color1 + lambdaFrac * color2;

    return color;
};

inline glm::vec3 tex2d_linear(const glm::vec2& imageTexCoord, unsigned char mipmap_index,
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
    glm::vec3 color;

    using cv::Vec4b;
    // int pixelIndex;
    // pixelIndex = getPixelIndex_wrap(x, y, texture->mipmaps[mipmapIndex].cols,
    // texture->mipmaps[mipmapIndex].rows);
    int pixelIndexCol = x;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].cols)
    {
        pixelIndexCol = 0;
    }
    int pixelIndexRow = y;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].rows)
    {
        pixelIndexRow = 0;
    }
    // std::cout << texture.mipmaps[mipmapIndex].cols << " " << texture.mipmaps[mipmapIndex].rows << " " <<
    // texture.mipmaps[mipmapIndex].channels() << std::endl;  cv::imwrite("mm.png",
    // texture.mipmaps[mipmapIndex]);
    color[0] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
    color[1] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
    color[2] = a * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x + 1, y, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x + 1;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].cols)
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].rows)
    {
        pixelIndexRow = 0;
    }
    color[0] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
    color[1] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
    color[2] += b * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x, y + 1, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].cols)
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y + 1;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].rows)
    {
        pixelIndexRow = 0;
    }
    color[0] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
    color[1] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
    color[2] += c * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

    // pixelIndex = getPixelIndex_wrap(x + 1, y + 1, texture.mipmaps[mipmapIndex].cols,
    // texture.mipmaps[mipmapIndex].rows);
    pixelIndexCol = x + 1;
    if (pixelIndexCol == texture.mipmaps[mipmap_index].cols)
    {
        pixelIndexCol = 0;
    }
    pixelIndexRow = y + 1;
    if (pixelIndexRow == texture.mipmaps[mipmap_index].rows)
    {
        pixelIndexRow = 0;
    }
    color[0] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[0];
    color[1] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[1];
    color[2] += d * texture.mipmaps[mipmap_index].at<Vec4b>(pixelIndexRow, pixelIndexCol)[2];

    return color;
};

} /* namespace detail */
} /* namespace render */
} /* namespace eos */

#endif /* TEXTURING_DETAIL_HPP_ */
