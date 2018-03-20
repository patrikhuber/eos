/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: python/pybind11_variant.hpp
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

#ifndef EOS_PYBIND11_VARIANT_HPP_
#define EOS_PYBIND11_VARIANT_HPP_

/**
 * @file python/pybind11_variant.hpp
 * @brief Define a type_caster for mpark::variant, which is used when the compiler doesn't have <variant> (e.g. on Apple).
 */

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)

#include "pybind11/stl.h"

#else

#include "eos/cpp17/variant.hpp"
#include "pybind11/stl.h"

namespace pybind11 {
namespace detail {

/**
 * @brief Type caster for mpark::variant, which is used when the compiler doesn't have <variant> (e.g. on Apple).
 */
template <typename... Ts>
struct type_caster<mpark::variant<Ts...>> : variant_caster<mpark::variant<Ts...>>
{
};

} /* namespace detail */
} /* namespace pybind11 */

#endif

#endif /* EOS_PYBIND11_VARIANT_HPP_ */
