/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/image/Pixel.hpp
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

#ifndef EOS_IMAGE_PIXEL_HPP
#define EOS_IMAGE_PIXEL_HPP

#include <array>
#include <type_traits>

namespace eos {
namespace core {

/**
 * @brief Represents a pixel with given type and number of channels.
 */
template <typename ElementType, int NumChannels>
class Pixel
{
public:
    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == NumChannels>>
    constexpr Pixel(Args... args) noexcept : data_{{static_cast<ElementType>(args)...}} {};

    constexpr explicit Pixel(const std::array<ElementType, NumChannels>& arr) noexcept : data_(arr){};

    constexpr ElementType& operator[](int i) noexcept
    {
        return data_[i];
    };
    constexpr const ElementType& operator[](int i) const noexcept
    {
        return data_[i];
    };

    const std::array<ElementType, NumChannels>& data() const noexcept
    {
        return data_;
    };

private:
    std::array<ElementType, NumChannels> data_;
};

template <typename ElementType, int NumChannels>
constexpr bool operator==(const Pixel<ElementType, NumChannels>& lhs,
                          const Pixel<ElementType, NumChannels>& rhs) noexcept
{
    return lhs.data() == rhs.data();
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_IMAGE_PIXEL_HPP */
