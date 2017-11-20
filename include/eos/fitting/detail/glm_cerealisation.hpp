/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/detail/glm_cerealisation.hpp
 *
 * Copyright 2016 Patrik Huber
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

#ifndef GLMCEREALISATION_HPP_
#define GLMCEREALISATION_HPP_

#include "cereal/cereal.hpp"

#include "glm/gtc/quaternion.hpp"

/**
 * @brief Serialisation of GLM \c glm::quat quaternion for the serialisation
 * library cereal (http://uscilab.github.io/cereal/index.html).
 *
 * Contains serialisation for \c glm::quat.
 */
namespace glm {

/**
 * @brief Serialisation of a glm::quat using cereal.
 *
 * @param[in] ar The archive to (de)serialise.
 * @param[in] vec The vector to (de)serialise.
 */
template <class Archive>
void serialize(Archive& ar, glm::quat& q)
{
    ar(q.w, q.x, q.y, q.z);
};

} /* namespace glm */

#endif /* GLMCEREALISATION_HPP_ */
