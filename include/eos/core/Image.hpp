/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Image.hpp
 *
 * Copyright 2017, 2018 Patrik Huber
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

#ifndef EOS_IMAGE_HPP
#define EOS_IMAGE_HPP

#include "eos/core/image/Pixel.hpp"
#include "eos/core/image/PixelTraits.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

namespace eos {
namespace core {

/**
 * @brief Class to represent images.
 *
 * The class currently uses row-major storage order.
 *
 * Some of the code is inspired by the Selene library (https://github.com/kmhofmann/selene, MIT license).
 */
template <class PixelType>
class Image
{
public:
    Image() = default;

    // Initialises with all zeros. This is a bit of an overhead. We probably have to use unique_ptr or a raw
    // pointer to get rid of this, as a vector can't have "uninitialised" (yet existing) elements.
    Image(int height, int width)
        : height_(height), width_(width), row_stride(PixelTraits<PixelType>::num_bytes * width)
    {
        // data_.reserve(row_stride * height); // just reserves, doesn't put any elements into the vector
        data_.resize(row_stride * height);
    };

    int height() const noexcept
    {
        return height_;
    };

    int width() const noexcept
    {
        return width_;
    };

    PixelType& operator()(int y, int x) noexcept
    {
        assert(data_.size() > 0); // byte_ptr() checks this but let's put the assert here too, because this
                                  // function is public-facing.
        assert(y >= 0 && y < height_);
        assert(x >= 0 && x < width_);
        return *data(y, x);
    };
    const PixelType& operator()(int y, int x) const noexcept
    {
        assert(data_.size() > 0); // byte_ptr() checks this but let's put the assert here too, because this
                                  // function is public-facing.
        assert(y >= 0 && y < height_);
        assert(x >= 0 && x < width_);
        return *data(y, x);
    };

private:
    int height_ = 0;
    int width_ = 0;

    int row_stride = 0; // In bytes. >1 means it's a row-major image, 1 would mean a col-major image.

    std::vector<std::uint8_t> data_; // Storage for the image data as bytes

    // Return a std::ptr_diff?
    int compute_data_offset(int y, int x) const noexcept
    {
        return row_stride * y + PixelTraits<PixelType>::num_bytes * x;
    };

    std::uint8_t* byte_ptr(int y, int x) noexcept
    {
        assert(data_.size() > 0);
        return &data_[0] + compute_data_offset(y, x);
    };
    const std::uint8_t* byte_ptr(int y, int x) const noexcept
    {
        assert(data_.size() > 0);
        return &data_[0] + compute_data_offset(y, x);
    };

    PixelType* data(int y, int x) noexcept
    {
        return reinterpret_cast<PixelType*>(byte_ptr(y, x));
    };
    const PixelType* data(int y, int x) const noexcept
    {
        return reinterpret_cast<const PixelType*>(byte_ptr(y, x));
    };
};

// Define type aliases for the most commonly used image types:
using Image1u = Image<std::uint8_t>;
using Image3u = Image<Pixel<std::uint8_t, 3>>;
using Image4u = Image<Pixel<std::uint8_t, 4>>;
using Image1f = Image<float>;
using Image3f = Image<Pixel<float, 3>>;
using Image1d = Image<double>;

} /* namespace core */
} /* namespace eos */

#endif /* EOS_IMAGE_HPP */
