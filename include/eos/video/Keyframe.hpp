/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/video/Keyframe.hpp
 *
 * Copyright 2016, 2017 Patrik Huber
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

#ifndef KEYFRAME_HPP_
#define KEYFRAME_HPP_

#include "eos/fitting/FittingResult.hpp"

namespace eos {
namespace video {

/**
 * @brief A keyframe selected by the fitting algorithm.
 *
 * Contains the original frame, all necessary fitting parameters, and a score.
 */
template <class ImageType>
struct Keyframe
{
    float score; // = 0.0f?
    ImageType frame;
    fitting::FittingResult fitting_result;
};

/**
 * @brief A keyframe selection that selects keyframes according to yaw pose and score.
 *
 * Separates the +-90° yaw pose range into 20° intervals (i.e. 90 to 70, ..., -10 to 10, ...), and puts frames
 * into each bin, until full. Replaces keyframes with better frames if the score is higher than that of
 * current keyframes.
 *
 * The yaw pose bins are currently hard-coded (9 bins, 20° intervals).
 */
template <class ImageType>
struct PoseBinningKeyframeSelector
{
public:
    PoseBinningKeyframeSelector(int frames_per_bin = 2) : frames_per_bin(frames_per_bin)
    {
        bins.resize(num_yaw_bins);
    };

    bool try_add(float frame_score, const ImageType& image, const fitting::FittingResult& fitting_result)
    {
        // Determine whether to add or not:
        auto yaw_angle = glm::degrees(glm::yaw(fitting_result.rendering_parameters.get_rotation()));
        auto idx = angle_to_index(yaw_angle);
        bool add_frame = false;
        if (bins[idx].size() < frames_per_bin) // always add when we don't have enough frames
            add_frame =
                true; // definitely adding - we wouldn't have to go through the for-loop on the next line.
        for (auto&& f : bins[idx])
        {
            if (frame_score > f.score)
                add_frame = true;
        }
        if (!add_frame)
        {
            return false;
        }
        // Add the keyframe:
        bins[idx].push_back(video::Keyframe{frame_score, image, fitting_result});
        if (bins[idx].size() > frames_per_bin)
        {
            // need to remove the lowest one:
            std::sort(std::begin(bins[idx]), std::end(bins[idx]),
                      [](const auto& lhs, const auto& rhs) { return lhs.score > rhs.score; });
            bins[idx].resize(frames_per_bin);
        }
        return true;
    };

    // Returns the keyframes as a vector.
    std::vector<Keyframe<ImageType>> get_keyframes() const
    {
        std::vector<Keyframe<ImageType>> keyframes;
        for (auto&& b : bins)
        {
            for (auto&& f : b)
            {
                keyframes.push_back(f);
            }
        }
        return keyframes;
    };

private:
    using BinContent = std::vector<Keyframe<ImageType>>;
    std::vector<BinContent> bins;
    const int num_yaw_bins = 9;
    int frames_per_bin;

    // Converts a given yaw angle to an index in the internal bins vector.
    // Assumes 9 bins and 20° intervals.
    static std::size_t angle_to_index(float yaw_angle)
    {
        if (yaw_angle <= -70.f)
            return 0;
        if (yaw_angle <= -50.f)
            return 1;
        if (yaw_angle <= -30.f)
            return 2;
        if (yaw_angle <= -10.f)
            return 3;
        if (yaw_angle <= 10.f)
            return 4;
        if (yaw_angle <= 30.f)
            return 5;
        if (yaw_angle <= 50.f)
            return 6;
        if (yaw_angle <= 70.f)
            return 7;
        return 8;
    };
};

} /* namespace video */
} /* namespace eos */

#endif /* KEYFRAME_HPP_ */
