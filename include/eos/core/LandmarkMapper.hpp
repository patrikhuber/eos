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

#include "boost/optional.hpp"

#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

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
class LandmarkMapper {
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
	 * @throws runtime_error if there is an error loading the mappings from the file.
	 */
	LandmarkMapper(std::string filename)
	{
		using std::string;
		using std::getline;

		std::ifstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error(string("LandmarkMapper: Could not open landmark mappings file: " + filename));
		}

		// We'll need these helper functions for the parsing:
		auto starts_with = [](const std::string& input, const std::string& match)
		{
			return input.size() >= match.size() && std::equal(match.begin(), match.end(), input.begin());
		};

		auto trim_left = [](const std::string& input, std::string pattern = " \t")
		{
			auto first = input.find_first_not_of(pattern);
			if (first == string::npos)
			{
				return input;
			}
			return input.substr(first, input.size());
		};

		// Read the actual file:
		string line;
		// Skip any comments (";") or empty lines at the beginning:
		while (getline(file, line)) {
			if (!starts_with(line, ";") && line != "")
			{
				// not a commented or not an empty line
				break;
			}
		}
		// First actual line should be "landmarkMappings"
		if (!starts_with(line, "landmarkMappings")) {
			throw std::runtime_error("LandmarkMapper error: First non-comment line should be \"landmarkMappings\".");
		}
		getline(file, line);
		// The next line has to be "{":
		if (line != "{") {
			throw std::runtime_error("LandmarkMapper error: Expected a \"{\" on the line following \"landmarkMappings\" statement.");
		}
		
		while (getline(file, line))
		{
			if (line == "}") { // end of the file
				break;
			}
			// on comment, continue:
			if (starts_with(trim_left(line), ";"))
			{
				continue;
			}
			std::stringstream line_stream(line);
			string lhs, rhs;
			if (!(line_stream >> lhs >> rhs)) {
				throw std::runtime_error(string("Landmark mappings format error while parsing the line: " + line));
			}
			landmark_mappings.insert(std::make_pair(lhs, rhs));
		}
		// We're at the end of the "landmarkMappings { ... }" block. Something else may follow, but we don't care.
	};

	/**
	 * @brief Converts the given landmark name to the mapped name.
	 *
	 * @param[in] landmark_name A landmark name to convert.
	 * @return The mapped landmark name if a mapping exists, an empty optional otherwise.
	 * @throws out_of_range exception if there is no mapping
	 *         for the given landmarkName.
	 */
	boost::optional<std::string> convert(std::string landmark_name) const
	{
		if (landmark_mappings.empty()) {
			// perform identity mapping, i.e. return the input
			return landmark_name;
		}
		else {
			auto&& converted_landmark = landmark_mappings.find(landmark_name);
			if (converted_landmark != std::end(landmark_mappings)) {
				// landmark mapping found, return it
				return converted_landmark->second;
			}
			else { // landmark_name does not match the key of any element in the map
				return boost::none;
			}
		}
	};

	/**
	 * @brief Returns the number of loaded landmark mappings.
	 *
	 * @return The number of landmark mappings.
	 */
	auto num_mappings() const
	{
		return landmark_mappings.size();
	};

private:
	std::map<std::string, std::string> landmark_mappings; ///< Mapping from one landmark name to a name in a different format.
};

	} /* namespace core */
} /* namespace eos */

#endif /* LANDMARKMAPPER_HPP_ */
