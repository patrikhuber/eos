/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: src/eos/core/LandmarkMapper.cpp
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
#include "eos/core/LandmarkMapper.hpp"

#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/info_parser.hpp"

#include <iostream>

using boost::property_tree::ptree;
using std::string;

namespace eos {
	namespace morphablemodel {

LandmarkMapper::LandmarkMapper(boost::filesystem::path filename)
{
	ptree configTree;
	try {
		boost::property_tree::info_parser::read_info(filename.string(), configTree);
	}
	catch (const boost::property_tree::ptree_error& error) {
		throw std::runtime_error(string("LandmarkMapper: Error reading landmark-mappings file: ") + error.what());
	}

	try {
		ptree ptLandmarkMappings = configTree.get_child("landmarkMappings");
		for (auto&& mapping : ptLandmarkMappings) {
			landmarkMappings.insert(make_pair(mapping.first, mapping.second.get_value<string>()));
		}
		std::cout << "Loaded a list of " << landmarkMappings.size() << " landmark mappings." << std::endl;
	}
	catch (const boost::property_tree::ptree_error& error) {
		throw std::runtime_error(string("LandmarkMapper: Error while parsing the mappings file: ") + error.what());
	}
	catch (const std::runtime_error& error) {
		throw std::runtime_error(string("LandmarkMapper: Error while parsing the mappings file: ") + error.what());
	}
}

string LandmarkMapper::convert(string landmarkName)
{
	if (landmarkMappings.empty()) {
		// perform identity mapping, i.e. return the input
		return landmarkName;
	}
	else {
		return landmarkMappings.at(landmarkName); // throws an out_of_range exception if landmarkName does not match the key of any element in the map
	}
}

	} /* namespace morphablemodel */
} /* namespace eos */
