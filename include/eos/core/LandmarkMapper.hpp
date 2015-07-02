/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/LandmarkMapper.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include <string>
#include <map>

namespace eos {
	namespace core {

/**
 * Represents a mapping from one kind of landmarks
 * to a different format. Mappings are stored in a
 * file (see share/ for an example for ibug landmarks).
 */
class LandmarkMapper {
public:
	/**
	 * Constructs a new landmark mapper that performs an identity mapping,
	 * that is, its output is the same as the input.
	 *
	 */
	LandmarkMapper() = default;

	/**
	 * Constructs a new landmark mapper from a mappings-file.
	 *
	 * @param[in] filename A file with landmark mappings.
	 * @throws runtime_error exception if there is an error
	 *         loading the mappings from the file.
	 */
	LandmarkMapper(boost::filesystem::path filename)
	{
		using std::string;
		using boost::property_tree::ptree;
		ptree configtree;
		try {
			boost::property_tree::info_parser::read_info(filename.string(), configtree);
		}
		catch (const boost::property_tree::ptree_error& error) {
			throw std::runtime_error(string("LandmarkMapper: Error reading landmark-mappings file: ") + error.what());
		}

		try {
			ptree pt_landmark_mappings = configtree.get_child("landmarkMappings");
			for (auto&& mapping : pt_landmark_mappings) {
				landmark_mappings.insert(make_pair(mapping.first, mapping.second.get_value<string>()));
			}
		}
		catch (const boost::property_tree::ptree_error& error) {
			throw std::runtime_error(string("LandmarkMapper: Error while parsing the mappings file: ") + error.what());
		}
		catch (const std::runtime_error& error) {
			throw std::runtime_error(string("LandmarkMapper: Error while parsing the mappings file: ") + error.what());
		}
	};

	/**
	 * Converts the given landmark name to the mapped name.
	 *
	 * @param[in] landmarkName A landmark name to convert.
	 * @return The mapped landmark name.
	 * @throws out_of_range exception if there is no mapping
	 *         for the given landmarkName.
	 */
	std::string convert(std::string landmark_name) const
	{
		if (landmark_mappings.empty()) {
			// perform identity mapping, i.e. return the input
			return landmark_name;
		}
		else {
			return landmark_mappings.at(landmark_name); // throws an out_of_range exception if landmarkName does not match the key of any element in the map
		}
	};

	/**
	 * Returns the number of loaded landmark mappings.
	 *
	 * @return The number of landmark mappings.
	 */
	auto size() const
	{
		return landmark_mappings.size();
	};

private:
	std::map<std::string, std::string> landmark_mappings; ///< Mapping from one landmark name to a name in a different format.
};

	} /* namespace core */
} /* namespace eos */

#endif /* LANDMARKMAPPER_HPP_ */
