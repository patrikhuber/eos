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

#include <string>
#include <map>

namespace eos {
	namespace morphablemodel {

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
	 */
	LandmarkMapper(boost::filesystem::path filename);

	/**
	 * Converts the given landmark name to the mapped name.
	 *
	 * @param[in] landmarkName A landmark name to convert.
	 * @return The mapped landmark name.
	 * @throws out_of_range exception if there is no mapping
	 *         for the given landmarkName.
	 */
	std::string convert(std::string landmarkName);

private:
	std::map<std::string, std::string> landmarkMappings; ///< Mapping from one landmark name to a name in a different format.
};

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* LANDMARKMAPPER_HPP_ */
