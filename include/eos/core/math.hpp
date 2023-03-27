/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/math.hpp
 *
 * Copyright 2023 Patrik Huber
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

#ifndef EOS_MATH_HPP
#define EOS_MATH_HPP

#include <limits>
#include <type_traits>

namespace eos {
namespace core {

/**
 * @brief Compile-time constant for pi (19 digits).
 */
template <class T>
constexpr T pi = T(3.1415926535897932385L);

/**
 * @brief Convert given degrees to radians.
 */
template <typename T>
T radians(T degrees)
{
    // Note: We may want to remove this assert, to allow ceres::Jet as type.
    static_assert(std::numeric_limits<T>::is_iec559, "radians() only accepts floating-point inputs.");
    return degrees * static_cast<T>(0.01745329251994329576923690768489);
};

/**
 * @brief Convert given radians to degree.
 */
template <typename T>
T degrees(T radians)
{
    // Note: We may want to remove this assert, to allow ceres::Jet as type.
    static_assert(std::numeric_limits<T>::is_iec559, "radians() only accepts floating-point inputs.");
    return radians * static_cast<T>(57.295779513082320876798154814105);
};

/**
 * @brief Returns the sign of the given number (-1, 0 or +1).
 *
 * According to StackOverflow, the < 0 part of the check triggers GCC's -Wtype-limits warning when
 * instantiated for an unsigned type. Thus we use two overloads, for signed and unsigned types.
 */
template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, int>::type inline constexpr sign(T const x)
{
    return T(0) < x;
};

/**
 * @brief Returns the sign of the given number (-1, 0 or +1).
 *
 * According to StackOverflow, the < 0 part of the check triggers GCC's -Wtype-limits warning when
 * instantiated for an unsigned type. Thus we use two overloads, for signed and unsigned types.
 */
template <typename T>
typename std::enable_if<std::is_signed<T>::value, int>::type inline constexpr sign(T const x)
{
    return (T(0) < x) - (x < T(0));
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_MATH_HPP */
