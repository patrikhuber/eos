/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/cpp17/detail/mpark_variant_serialization.hpp
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

#ifndef EOS_MPARK_VARIANT_SERIALIZATION_HPP_
#define EOS_MPARK_VARIANT_SERIALIZATION_HPP_

#include "eos/cpp17/detail/mpark_variant.hpp"

#include "cereal/cereal.hpp"

#include <type_traits>
#include <cstdint>

namespace cereal {
namespace variant_detail {

//! @internal
template <class Archive>
struct variant_save_visitor
{
    variant_save_visitor(Archive& ar_) : ar(ar_) {}

    template <class T>
    void operator()(T const& value) const
    {
        ar(CEREAL_NVP_("data", value));
    }

    Archive& ar;
};

//! @internal
template <int N, class Variant, class... Args, class Archive>
typename std::enable_if<N == mpark::variant_size_v<Variant>, void>::type
load_variant(Archive& /*ar*/, int /*target*/, Variant& /*variant*/)
{
    throw ::cereal::Exception("Error traversing variant during load");
}

//! @internal
template <int N, class Variant, class H, class... T, class Archive>
    typename std::enable_if <
    N<mpark::variant_size_v<Variant>, void>::type load_variant(Archive& ar, int target, Variant& variant)
{
    if (N == target)
    {
        H value;
        ar(CEREAL_NVP_("data", value));
        variant = std::move(value);
    } else
        load_variant<N + 1, Variant, T...>(ar, target, variant);
}

} // namespace variant_detail

//! Saving for mpark::variant
template <class Archive, typename VariantType1, typename... VariantTypes>
inline void CEREAL_SAVE_FUNCTION_NAME(Archive& ar,
                                      mpark::variant<VariantType1, VariantTypes...> const& variant)
{
    std::int32_t index = static_cast<std::int32_t>(variant.index());
    ar(CEREAL_NVP_("index", index));
    variant_detail::variant_save_visitor<Archive> visitor(ar);
    std::visit(visitor, variant);
}

//! Loading for mpark::variant
template <class Archive, typename... VariantTypes>
inline void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, mpark::variant<VariantTypes...>& variant)
{
    using variant_t = typename mpark::variant<VariantTypes...>;

    std::int32_t index;
    ar(CEREAL_NVP_("index", index));
    if (index >= static_cast<std::int32_t>(mpark::variant_size_v<variant_t>))
        throw Exception("Invalid 'index' selector when deserializing mpark::variant");

    variant_detail::load_variant<0, variant_t, VariantTypes...>(ar, index, variant);
}

} // namespace cereal

#endif /* EOS_MPARK_VARIANT_SERIALIZATION_HPP_ */
