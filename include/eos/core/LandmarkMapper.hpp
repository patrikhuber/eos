/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/LandmarkMapper.hpp
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

#ifndef LANDMARKMAPPER_HPP_
#define LANDMARKMAPPER_HPP_

#include "eos/cpp17/optional.hpp"

#include "toml.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace eos {
namespace core {

/**
 * @brief Represents a mapping from one kind of landmarks
 * to a different format (e.g. model vertices).
 *
 * When fitting the 3D model to an image, a correspondence must
 * be known from the 2D image landmarks to 3D vertex points in
 * the Morphable Model. The 3D model defines all its points in
 * the form of vertex ids.
 * These mappings are stored in a file, see the \c share/ folder for
 * an example for mapping 2D ibug landmarks to 3D model vertex indices.
 *
 * The LandmarkMapper thus has two main use cases:
 * - Mapping 2D landmark points to 3D vertices
 * - Converting one set of 2D landmarks into another set of 2D
 *   landmarks with different identifiers.
 */
class LandmarkMapper
{
public:
    /**
     * @brief Constructs a new landmark mapper that performs an identity
     * mapping, that is, its output is the same as the input.
     */
    LandmarkMapper() = default;

    /**
     * @brief Constructs a new landmark mapper from a file containing
     * mappings from one set of landmark identifiers to another.
     *
     * In case the file contains no mappings, a landmark mapper
     * that performs an identity mapping is constructed.
     *
     * @param[in] filename A file with landmark mappings.
     * @throws runtime_error or toml::exception if there is an error loading the mappings from the file.
     */
    LandmarkMapper(std::string filename)
    {
        // parse() as well as extracting the data can throw std::runtime error or toml::exception,
        // so ideally you'd want to call this c'tor within a try-catch.
        const auto data = toml::parse(filename);

        const auto mappings_table = toml::get<toml::Table>(data.at("landmark_mappings"));
        // The key in the config is always a string. The value however may be a string or an integer, so we
        // check for that and convert to a string.
        for (const auto& mapping : mappings_table)
        {
            std::string f = mapping.first;
            std::string value;
            switch (mapping.second.type())
            {
            case toml::value_t::Integer:
                value = std::to_string(toml::get<int>(mapping.second));
                break;
            case toml::value_t::String:
                value = toml::get<std::string>(mapping.second);
                break;
            default:
                throw std::runtime_error("unexpected type : " + toml::stringize(mapping.second.type()));
            }

            landmark_mappings.emplace(mapping.first, value);
        }
    };

    /**
     * @brief Constructs a new landmark mapper from a set of existing mappings.
     *
     * @param[in] mappings A set of landmark mappings.
     */
    LandmarkMapper(const std::unordered_map<std::string, std::string>& mappings) : landmark_mappings(mappings)
    {};

    /**
     * @brief Converts the given landmark name to the mapped name.
     *
     * In the case the mapper is empty (no mappings defined at all), the mapper will perform
     * an identity mapping and return the \p landmark_name that was input.
     *
     * @param[in] landmark_name A landmark name to convert.
     * @return The mapped landmark name if a mapping exists, an empty optional otherwise.
     */
    cpp17::optional<std::string> convert(std::string landmark_name) const
    {
        if (landmark_mappings.empty())
        {
            // perform identity mapping, i.e. return the input
            return landmark_name;
        }

        const auto& converted_landmark = landmark_mappings.find(landmark_name);
        if (converted_landmark != std::end(landmark_mappings))
        {
            // landmark mapping found, return it
            return converted_landmark->second;
        } else
        { // landmark_name does not match the key of any element in the map
            return cpp17::nullopt;
        }
    };

    /**
     * @brief Returns the number of loaded landmark mappings.
     *
     * @return The number of landmark mappings.
     */
    auto num_mappings() const {
        return landmark_mappings.size();
    };

    /**
     * @brief Returns the mappings held by this mapper.
     *
     * @return All mappings contained in the mapper.
     */
    const auto& get_mappings() const {
        return landmark_mappings;
    };

private:
    std::unordered_map<std::string, std::string> landmark_mappings; ///< Mapping from one
                                                                    ///< landmark name to a name
                                                                    ///< in a different format.
};

} /* namespace core */
} /* namespace eos */

#endif /* LANDMARKMAPPER_HPP_ */
