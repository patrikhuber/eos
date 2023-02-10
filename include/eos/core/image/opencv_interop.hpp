/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/image/opencv_interop.hpp
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

#ifndef EOS_IMAGE_OPENCV_INTEROP_HPP
#define EOS_IMAGE_OPENCV_INTEROP_HPP

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
    cv::Mat opencv_matrix(static_cast<int>(image.height()), static_cast<int>(image.width()), CV_8UC4);
    for (int c = 0; c < image.width(); ++c)
    { // size_t
        for (int r = 0; r < image.height(); ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<cv::Vec4b>(r, c) =
                cv::Vec4b(image(r, c)[0], image(r, c)[1], image(r, c)[2], image(r, c)[3]);
        }
    }
    return opencv_matrix;
};

inline cv::Mat to_mat(const Image3u& image)
{
    cv::Mat opencv_matrix(image.height(), image.width(), CV_8UC3);
    for (int c = 0; c < image.width(); ++c)
    { // size_t
        for (int r = 0; r < image.height(); ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<cv::Vec3b>(r, c) = cv::Vec3b(image(r, c)[0], image(r, c)[1], image(r, c)[2]);
        }
    }
    return opencv_matrix;
};

inline cv::Mat to_mat(const Image1d& image)
{
    cv::Mat opencv_matrix(static_cast<int>(image.height()), static_cast<int>(image.width()), CV_64FC1);
    for (int c = 0; c < image.width(); ++c)
    { // size_t
        for (int r = 0; r < image.height(); ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<double>(r, c) = image(r, c);
        }
    }
    return opencv_matrix;
};

inline cv::Mat to_mat(const Image1u& image)
{
    cv::Mat opencv_matrix(static_cast<int>(image.height()), static_cast<int>(image.width()), CV_8UC1);
    for (int c = 0; c < image.width(); ++c)
    { // size_t
        for (int r = 0; r < image.height(); ++r)
        {
            // auto vals = image(r, c);
            opencv_matrix.at<unsigned char>(r, c) = image(r, c);
        }
    }
    return opencv_matrix;
};

/**
 * Returns an Image3u from a given cv::Mat with type CV_8UC3. The channel order is not changed, i.e. if the
 * cv::Mat is BGR, the output Image3u will have BGR channel ordering too.
 */
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
            converted(r, c) = {image.at<cv::Vec3b>(r, c)[0], image.at<cv::Vec3b>(r, c)[1],
                               image.at<cv::Vec3b>(r, c)[2]};
        }
    }
    return converted;
};

/**
 * Supports both CV_8UC3 and CV_8UC4 cv::Mat as input images. If a CV_8UC3 images is given, then all pixels of
 * the alpha channel of the returned image are set to 255.
 */
inline Image4u from_mat_with_alpha(const cv::Mat& image)
{
    if (image.type() != CV_8UC3 && image.type() != CV_8UC4)
    {
        throw std::runtime_error("Can only convert a CV_8UC3 or CV_8UC4 cv::Mat to an eos::core::Image4u.");
    }

    Image4u converted(image.rows, image.cols);
    if (image.type() == CV_8UC3)
    {
        for (int r = 0; r < image.rows; ++r)
        {
            for (int c = 0; c < image.cols; ++c)
            {
                converted(r, c) = {image.at<cv::Vec3b>(r, c)[0], image.at<cv::Vec3b>(r, c)[1],
                                   image.at<cv::Vec3b>(r, c)[2], 255};
            }
        }
    } else if (image.type() == CV_8UC4)
    {
        for (int r = 0; r < image.rows; ++r)
        {
            for (int c = 0; c < image.cols; ++c)
            {
                converted(r, c) = {image.at<cv::Vec4b>(r, c)[0], image.at<cv::Vec4b>(r, c)[1],
                                   image.at<cv::Vec4b>(r, c)[2], image.at<cv::Vec4b>(r, c)[3]};
            }
        }
    }
    return converted;
};

} /* namespace core */
} /* namespace eos */

#endif /* EOS_IMAGE_OPENCV_INTEROP_HPP */
