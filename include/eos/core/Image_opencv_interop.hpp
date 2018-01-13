/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/Image_opencv_interop.hpp
 *
 * Copyright 2017 Patrik Huber
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

#ifndef IMAGE_OPENCV_INTEROP_HPP_
#define IMAGE_OPENCV_INTEROP_HPP_

#include "eos/core/Image.hpp"

#include "opencv2/core/core.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace eos {
namespace core {

// We can support different types by making this a template and constexpr if? :-)
inline cv::Mat to_mat(const Image4u& image)
{
    cv::Mat opencv_matrix(image.rows, image.cols, CV_8UC4);
    for (int c = 0; c < image.cols; ++c)
    { // size_t
        for (int r = 0; r < image.rows; ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<cv::Vec4b>(r, c) =
                cv::Vec4b(image(r, c)[0], image(r, c)[1], image(r, c)[2], image(r, c)[3]);
        }
    }
    return opencv_matrix;
};

inline cv::Mat to_mat(const Image1d& image)
{
    cv::Mat opencv_matrix(image.rows, image.cols, CV_64FC1);
    for (int c = 0; c < image.cols; ++c)
    { // size_t
        for (int r = 0; r < image.rows; ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<double>(r, c) = image(r, c);
        }
    }
    return opencv_matrix;
};

inline cv::Mat to_mat(const Image1u& image)
{
    cv::Mat opencv_matrix(image.rows, image.cols, CV_8UC1);
    for (int c = 0; c < image.cols; ++c)
    { // size_t
        for (int r = 0; r < image.rows; ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<unsigned char>(r, c) = image(r, c);
        }
    }
    return opencv_matrix;
};

inline Image3u from_mat(const cv::Mat& image)
{
    if (image.type() != CV_8UC3)
    {
        throw std::runtime_error("Can only convert a CV_8UC3 cv::Mat to an eos::core::Image3u.");
    }

    Image3u converted(image.rows, image.cols);
    for (int r = 0; r < image.rows; ++r)
    {
        for (int c = 0; c < image.cols; ++c)
        {
            converted(r, c) = std::array<std::uint8_t, 3>{
                image.at<cv::Vec3b>(r, c)[0], image.at<cv::Vec3b>(r, c)[1], image.at<cv::Vec3b>(r, c)[2]};
        }
    }
    return converted;
};

} /* namespace core */
} /* namespace eos */

#endif /* IMAGE_OPENCV_INTEROP_HPP_ */
