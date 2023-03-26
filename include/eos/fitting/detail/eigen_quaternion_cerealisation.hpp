/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/detail/eigen_quaternion_cerealisation.hpp
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

#ifndef EOS_EIGEN_QUATERNION_CEREALISATION_HPP
#define EOS_EIGEN_QUATERNION_CEREALISATION_HPP

#include "cereal/cereal.hpp"

#include "Eigen/Core"

/**
 * @brief Serialisation of Eigen's \c Eigen::Quaternion for the serialisation library cereal
 * (http://uscilab.github.io/cereal/index.html).
 *
 * Contains serialisation for \c Eigen::Quaternion.
 */
namespace Eigen {

/**
 * @brief Serialisation of a Eigen::Quaternion using cereal.
 *
 * Valid ScalarType's are float and double (Quaternionf and Quaterniond).
 *
 * @param[in] ar The archive to (de)serialise.
 * @param[in] q The quaternion to (de)serialise.
 */
template <class Archive, class ScalarType>
void serialize(Archive& ar, Eigen::Quaternion<ScalarType>& q)
{
    ar(q.w(), q.x(), q.y(), q.z());
};

} /* namespace Eigen */

#endif /* EOS_EIGEN_QUATERNION_CEREALISATION_HPP */
