/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/cpp17/detail/akrzemi1_optional_serialization.hpp
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

#ifndef EOS_AKRZEMI1_OPTIONAL_SERIALIZATION_HPP_
#define EOS_AKRZEMI1_OPTIONAL_SERIALIZATION_HPP_

#include "eos/cpp17/detail/akrzemi1_optional.hpp"

#include "cereal/cereal.hpp"

namespace cereal {

//! Saving for akrzemi1::optional
template <class Archive, typename T>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const akrzemi1::optional<T>& optional)
{
    if (!optional)
    {
        ar(CEREAL_NVP_("nullopt", true));
    } else
    {
        ar(CEREAL_NVP_("nullopt", false), CEREAL_NVP_("data", *optional));
    }
}

//! Loading for akrzemi1::optional
template <class Archive, typename T>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, akrzemi1::optional<T>& optional)
{
    bool nullopt;
    ar(CEREAL_NVP_("nullopt", nullopt));

    if (nullopt)
    {
        optional = akrzemi1::nullopt;
    } else
    {
        T value;
        ar(CEREAL_NVP_("data", value));
        optional = std::move(value);
    }
}

} // namespace cereal

#endif /* EOS_AKRZEMI1_OPTIONAL_SERIALIZATION_HPP_ */
