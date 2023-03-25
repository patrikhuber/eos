/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/rotation_angles.hpp
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

#ifndef EOS_FITTING_ROTATION_ANGLES_HPP
#define EOS_FITTING_ROTATION_ANGLES_HPP

#include "eos/core/math.hpp"

#include "Eigen/Core"

#include <cmath>

namespace eos {
namespace fitting {

/**
 * @brief Computes Tait-Bryan angles (in radians) from the given rotation matrix and axes order.
 *
 * Calls Eigen::Matrix3::eulerAngles(), but then swap the solution for the one where the middle (pitch) axis
 * is constrained to -PI/2 to PI/2.
 */
template <typename T>
inline Eigen::Matrix<T, 3, 1> tait_bryan_angles(Eigen::Matrix<T, 3, 3> rotation_matrix, Eigen::Index axis_0,
                                                Eigen::Index axis_1, Eigen::Index axis_2)
{
    Eigen::Matrix<T, 3, 1> euler_angles = rotation_matrix.eulerAngles(axis_0, axis_1, axis_2);
    // Eigen::Matrix3X.eulerAngles() returns the solution where the first axis is constrained from 0 to PI.
    // This is not what we usually want in robotics; we want the other solution (there are two in the general
    // case) where the middle (pitch) axis is constrained to -PI/2 to PI/2. See
    // https://gitlab.com/libeigen/eigen/-/issues/2617#note_1298729055
    if (std::abs(euler_angles(1)) > T(core::pi<T> / 2.0))
    {
        euler_angles.array() -= T(core::pi<T>) * euler_angles.array().sign();
        euler_angles(1) = -euler_angles(1);
    }
    return euler_angles;
};

} /* namespace fitting */
} /* namespace eos */

#endif /* EOS_FITTING_ROTATION_ANGLES_HPP */
