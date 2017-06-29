/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: python/pybind11_Image.hpp
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

#include "pybind11/numpy.h"

#include "Eigen/Core"

#include <cstddef>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/**
 * @file python/pybind11_Image.hpp
 * @brief Transparent conversion to and from Python for eos::core::Image.
 *
 * Numpy uses row-major storage order by default.
 * eos::core::Image uses col-major storage (like Eigen).
 * 
 * If given non-standard strides or something from numpy, probably doesn't work.
 * May need to .clone()? in numpy before passing to the C++ function.
 */


/**
 * @brief Transparent conversion for eos::core::Image to and from Python.
 *
 * Converts eos::core::Image's to and from Python. Can construct a eos::core::Image from numpy arrays,
 * as well as potentially other Python array types.
 *
 * //- Supports only contiguous matrices
 * //- The numpy array has to be in default (row-major) storage order
 * //- Non-default strides are not implemented.
 */
template<class T, int num_channels>
struct type_caster<eos::core::Image<T, num_channels>>
{
	using image_type = eos::core::Image<T, num_channels>;
	//using Scalar = T;
	//static constexpr std::size_t num_elements = N;
	
	bool load(handle src, bool)
	{
		// Since cv::Mat has its time dynamically specified at run-time, we can't bind functions
		// that take a cv::Mat to any particular Scalar type.
		// Thus the data we get from python can be any type.
		auto buf = pybind11::array::ensure(src);
		if (!buf)
			return false;

		// Todo: We should probably check that buf.strides(i) is "default", by dividing it by the Scalar type or something.

		if (!pybind11::isinstance<pybind11::array_t<std::uint8_t>>(buf))
		{
			return false; // we only convert uint8_t for now.
		}

		if (buf.ndim() != 3) {
			return false;
		}
		// We got something with 3 dimensions, i.e. an image with 2, 3 or 4 channels (or 'k' for that matter):
		auto num_chans = buf.shape(2); // Check whether 3 or 4 and abort otherwise??
		auto shp = buf.shape();
		auto str = buf.strides();
		// Now: If our Image class had support for col/row major, we could just map buf.mutable_data().
		// Like with OpenCV: value = cv::Mat(buf.shape(0), buf.shape(1), opencv_type, buf.mutable_data());
		// But since it doesn't, we just copy for now:
		value = core::Image3u(buf.shape(0), buf.shape(1));
		array_t<std::uint8_t> test(buf);
		for (int r = 0; r < buf.shape(0); ++r) {
			for (int c = 0; c < buf.shape(1); ++c) {
				value(r, c)[0] = test.at(r, c, 0);
				value(r, c)[1] = test.at(r, c, 1);
				value(r, c)[2] = test.at(r, c, 2);
			}
		}
		return true;
	};

    static handle cast(const image_type& src, return_value_policy /* policy */, handle /* parent */)
	{

		//const auto opencv_depth = src.depth(); // CV_8U etc - the element type
		const auto num_chans = 4;
		std::vector<std::size_t> shape;
		if (num_chans == 1)
		{
			shape = { (size_t)src.rows, (size_t)src.cols };
			// if either of them is == 1, we could specify only 1 value for shape - but be careful with strides,
			// if there's a col-vector, I don't think we can do it without using strides.
			// Also, check what happens in python when we pass a col & row vec respectively.
		}
		else if (num_chans == 2 || num_chans == 3 || num_chans == 4)
		{
			shape = { (size_t)src.rows, (size_t)src.cols, (size_t)num_chans };
		}
		else {
			throw std::runtime_error("Cannot return matrices with more than 4 channels back to Python.");
			// We could probably implement this quite easily but >4 channel images/matrices don't occur often.
		}

		// Now return the data, depending on its type:
		//if (opencv_depth == CV_8U) // if element type = uint8_t...
		//{
			// (2048, 4, 1) is default which results in transposed image
			std::vector<size_t> strides = { 4, 2048, 1 }; // This works now. In numpy the strides are (2048, 4, 1) though. I think a copy gets created nevertheless?
			return array(pybind11::dtype::of<std::uint8_t>(), shape, strides, &src.data[0]).release();
		//}
		//else {
		//	throw std::runtime_error("Can currently only return matrices of type 8U back to Python. Other types can be added if needed.");
		//}
	};

    PYBIND11_TYPE_CASTER(image_type, _("numpy.ndarray[uint8|int32|float32|float64[m, n, d]]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
