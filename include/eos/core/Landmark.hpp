/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Landmark.hpp
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

#ifndef LANDMARK_HPP_
#define LANDMARK_HPP_

#include <string>
#include <vector>

namespace eos {
	namespace core {

/**
 * Representation of a landmark, consisting of a landmark name and
 * coordinates of the given type. Usually, the type would be cv::Vec2f.
 */
template<class LandmarkType>
struct Landmark
{
	std::string name;
	LandmarkType coordinates;
};

/**
 * A trivial collection of landmarks that somehow belong together.
 */
template<class LandmarkType> using LandmarkCollection = std::vector<Landmark<LandmarkType>>;

/**
 * Filters the given LandmarkCollection and returns a new LandmarkCollection
 * containing all landmarks whose name matches the one given by \p filter.
 *
 * @param[in] landmarks The input LandmarkCollection to be filtered.
 * @param[in] filter A landmark name (identifier) by which the given LandmarkCollection is filtered.
 * @return A new, filtered LandmarkCollection.
 */
template<class T>
LandmarkCollection<T> filter(const LandmarkCollection<T>& landmarks, const std::vector<std::string>& filter)
{
	LandmarkCollection<T> filtered_landmarks;
	using std::begin;
	using std::end;
	std::copy_if(begin(landmarks), end(landmarks), std::back_inserter(filtered_landmarks),
		[&](const Landmark<T>& lm) { return std::find(begin(filter), end(filter), lm.name) != end(filter); }
	);
	return filtered_landmarks;
};

	} /* namespace core */
} /* namespace eos */

#endif /* LANDMARK_HPP_ */
