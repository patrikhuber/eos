/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/read_pts_landmarks.hpp
 *
 * Copyright 2014, 2017 Patrik Huber
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

#ifndef EOS_READ_PTS_LANDMARKS_HPP
#define EOS_READ_PTS_LANDMARKS_HPP

#include "eos/core/Landmark.hpp"

#include "Eigen/Core"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

namespace eos {
namespace core {

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
inline LandmarkCollection<Eigen::Vector2f> read_pts_landmarks(std::string filename)
{
    using Eigen::Vector2f;
    using std::getline;
    using std::string;
    LandmarkCollection<Vector2f> landmarks;
    landmarks.reserve(68);

    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error(string("Could not open landmark file: " + filename));
    }

    string line;
    // Skip the first 3 lines, they're header lines:
    getline(file, line); // 'version: 1'
    getline(file, line); // 'n_points : 68'
    getline(file, line); // '{'

    int ibugId = 1;
    while (getline(file, line))
    {
        if (line == "}")
        { // end of the file
            break;
        }
        std::stringstream lineStream(line);

        Landmark<Vector2f> landmark;
        landmark.name = std::to_string(ibugId);
        if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1]))
        {
            throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
        }
        // From the iBug website:
        // "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1
        // being the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
        // ==> So we shift every point by 1:
        landmark.coordinates[0] -= 1.0f;
        landmark.coordinates[1] -= 1.0f;
        landmarks.emplace_back(landmark);
        ++ibugId;
    }
    return landmarks;
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_READ_PTS_LANDMARKS_HPP */
