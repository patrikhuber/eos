/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/include/mexplus_opencv.hpp
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

#ifndef MEXPLUS_OPENCV_HPP_
#define MEXPLUS_OPENCV_HPP_

#include "mexplus/mxarray.h"

#include "opencv2/core/core.hpp"

#include "mex.h"

#include <cstdint>
#include <vector>

namespace mexplus {

/**
 * @brief Convert a cv::Mat to a Matlab matrix.
 *
 * Conversion should work for matrices of any type, with any number of channels.
 * The function creates a new MxArray of the same data type and copies the data over.
 *
 * Note: Even non-standard step/stride sizes should work, but that is not tested.
 */
template <typename InputScalar, typename OutputScalar>
void deepcopy_and_transpose(const cv::Mat& in, MxArray& out)
{
    const std::size_t num_rows = in.rows;
    const std::size_t num_cols = in.cols;
    const std::size_t num_chans = in.channels();
    out = MxArray::Numeric<OutputScalar>({num_rows, num_cols, num_chans});

    for (std::size_t c = 0; c < num_cols; ++c)
    { // outer loop over rows would be faster if OpenCV stores data row-major?
        for (std::size_t r = 0; r < num_rows; ++r)
        {
            for (std::size_t chan = 0; chan < num_chans; ++chan)
            {
                out.set(std::vector<mwIndex>{r, c, chan}, in.ptr<InputScalar>(r, c)[chan]);
            }
        }
    }
};

// Note/Todo: Currently only works for 3-chan and 4-chan U8 images
// Matlab stores matrices in col - major order in memory, OpenCV stores them in row - major.Thus, we copy & transpose...
template <typename InputScalar, typename OutputScalar>
void deepcopy_and_transpose(const MxArray& in, cv::Mat& out)
{
    auto r = in.rows();
    auto c = in.cols(); // if dimensionSize() == 3, this returns cols * third_dim
    auto d = in.dimensions();
    auto ds = in.dimensionSize();
    auto e = in.elementSize(); // Number of bytes required to store each data element
    auto s = in.size();
    auto n = in.isNumeric();

    const auto num_channels = [ num_dims = in.dimensionSize(), dims = in.dimensions() ]()
    {
        if (num_dims == 1 || num_dims == 2)
        {
            return std::size_t(1);
        } else if (num_dims == 3)
        { // the dims vector has 3 entries, the 3rd entry tell us if there are 3 or 4 channels.
            return dims[2];
        }
    }
    ();

    const auto actual_cols = [&num_channels, num_cols = in.cols() ]()
    {
        if (num_channels == 3 || num_channels == 4) // aargh... simplify all this... just use in.dimensions()?
        {
            return num_cols / num_channels;
        } else
        {
            return num_cols;
        }
    }
    ();

    std::vector<cv::Mat> channels;
    for (size_t c = 0; c < num_channels; ++c)
    {
        cv::Mat outmat;
        // Note: I think this doesn't actually copy the data, does it?
        // We construct with (cols, rows) because it'll later get transposed.
        cv::Mat inmat(actual_cols, in.rows(), cv::DataType<InputScalar>::type,
                      static_cast<void*>(
                          const_cast<InputScalar*>(in.getData<InputScalar>() + actual_cols * in.rows() * c)));
        inmat.convertTo(outmat, cv::DataType<OutputScalar>::type);
        cv::transpose(outmat, outmat);
        channels.push_back(outmat);
    }
    cv::merge(channels, out);
};

/**
 * @brief Convert a cv::Mat to a Matlab matrix.
 *
 * Conversion should work for matrices of type CV_8U, CV_32F and CV_64F, with any number of channels.
 */
template <>
mxArray* MxArray::from(const cv::Mat& opencv_matrix)
{
    MxArray out_array;
    if (opencv_matrix.depth() == CV_8U)
    {
        deepcopy_and_transpose<std::uint8_t, std::uint8_t>(opencv_matrix, out_array);
    } else if (opencv_matrix.depth() == CV_32F)
    {
        deepcopy_and_transpose<float, float>(opencv_matrix, out_array);
    } else if (opencv_matrix.depth() == CV_64F)
    {
        deepcopy_and_transpose<double, double>(opencv_matrix, out_array);
    } else
    {
        mexErrMsgIdAndTxt("eos:matlab", "Can only convert CV_8U, CV_32F and CV_64F matrices at this point.");
    }
    return out_array.release();
};

/**
 * @brief Convert a 3-channel or 4-channel uint8 Matlab matrix to a cv::Mat.
 *
 * Currently only works for 3-channel and 4-channel uint8 Matlab matrices (i.e. images and isomaps).
 * Todo: Add some error detection, it will silently fail or crash when the type/channels are
 * not correct.
 */
template <>
void MxArray::to(const mxArray* in_array, cv::Mat* opencv_matrix)
{
    MxArray array(in_array);
    deepcopy_and_transpose<std::uint8_t, std::uint8_t>(array, *opencv_matrix);
};

} /* namespace mexplus */

#endif /* MEXPLUS_OPENCV_HPP_ */
