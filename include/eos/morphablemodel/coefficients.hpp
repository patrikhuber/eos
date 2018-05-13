/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/coefficients.hpp
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

#ifndef EOS_COEFFICIENTS_HPP
#define EOS_COEFFICIENTS_HPP

#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"

#include <string>
#include <vector>
#include <fstream>

namespace eos {
namespace morphablemodel {

/**
 * Saves coefficients (for example PCA shape coefficients) to a json file.
 *
 * @param[in] coefficients A vector of coefficients.
 * @param[in] filename The file to write.
 * @throws std::runtime_error if unable to open the given file for writing.
 */
inline void save_coefficients(std::vector<float> coefficients, std::string filename)
{
    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Error opening file for writing: " + filename);
    }
    cereal::JSONOutputArchive output_archive(file);
    output_archive(cereal::make_nvp("shape_coefficients", coefficients));
};

/**
 * Loads coefficients (for example PCA shape coefficients) from a json file.
 *
 * @param[in] filename The file to write.
 * @return Returns vector of floats.
 * @throws std::runtime_error if unable to open the given file for reading.
 */
inline std::vector<float> load_coefficients(std::string filename)
{
    std::vector<float> coefficients;
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Error opening file for reading: " + filename);
    }
    cereal::JSONInputArchive input_archive(file);
    input_archive(cereal::make_nvp("shape_coefficients", coefficients));
    return coefficients;
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* EOS_COEFFICIENTS_HPP */
