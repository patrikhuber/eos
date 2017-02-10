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

// Note/Todo: Currently only works for 4-chan U8 images
template <typename InputScalar, typename OutputScalar>
void deepcopy_and_transpose(const cv::Mat& in, MxArray& out)
{
    if (in.channels() != 4)
    {
        mexErrMsgIdAndTxt("eos:matlab", "Currently only works for 4-channel U8 images.");
    }

    const std::size_t num_rows = in.rows;
    const std::size_t num_cols = in.cols;
    const std::size_t num_chans = in.channels();
    out = MxArray::Numeric<std::uint8_t>({num_rows, num_cols, num_chans});

    for (std::size_t c = 0; c < num_cols; ++c) {
        for (std::size_t r = 0; r < num_rows; ++r) {
            for (std::size_t chan = 0; chan < num_chans; ++chan) {
                out.set(std::vector<mwIndex>{r, c, chan}, in.at<cv::Vec4b>(r, c)[chan]);
            }
        }
    }
};

// Note/Todo: Currently only works for 3-chan U8 images
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
	
	const auto num_channels = [num_dims = in.dimensionSize()]() {
		if (num_dims == 1 || num_dims == 2)
		{
			return 1;
		}
		else if (num_dims == 3) {
			return 3;
		}
		// What about 4-channel images, what dimensionSize() will they have?
	}();

	const auto actual_cols = [&num_channels, num_cols = in.cols()]() {
		if (num_channels == 3)
		{
			return num_cols / num_channels;
		}
		else {
			return num_cols;
		}
	}();

    std::vector<cv::Mat> channels;
    for (size_t c = 0; c < num_channels; ++c)
    {
        cv::Mat outmat;
		// Note: I think this doesn't actually copy the data, does it?
        cv::Mat inmat(actual_cols, in.rows(), cv::DataType<InputScalar>::type,
            static_cast<void*>(const_cast<InputScalar*>(in.getData<InputScalar>() + actual_cols * in.rows() * c)));
        inmat.convertTo(outmat, cv::DataType<OutputScalar>::type);
        cv::transpose(outmat, outmat);
        channels.push_back(outmat);
    }
    cv::merge(channels, out);
};

/**
 * @brief Convert a 4-channel uchar cv::Mat to a Matlab matrix.
 *
 * Currently, only CV_8UC4 matrices are supported. Don't try to convert anything else.
 * Todo: Add some error detection, it will silently fail or crash when the type/channels are
 * not correct.
 */
template <>
mxArray* MxArray::from(const cv::Mat& opencv_matrix)
{
    MxArray out_array;
    deepcopy_and_transpose<std::uint8_t, std::uint8_t>(opencv_matrix, out_array);
    return out_array.release();
};

/**
 * @brief Convert a 3-channel uint8 Matlab matrix to a cv::Mat.
 *
 * Currently only works for 3-channel uint8 Matlab matrices (i.e. images)!
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
