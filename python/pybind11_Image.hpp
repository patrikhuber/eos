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

#ifndef EOS_PYBIND11_IMAGE_HPP_
#define EOS_PYBIND11_IMAGE_HPP_

#include "pybind11/numpy.h"

#include "Eigen/Core"

#include <cstddef>
#include <vector>

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
 * @brief Transparent conversion for eos::core::Image3u to and from Python.
 *
 * Converts an eos::core::Image3u to and from Python. Can construct a eos::core::Image3u from numpy arrays,
 * as well as potentially other Python array types.
 *
 * Note: Not sure what happens if the given numpy array is not contiguous, not in default (row-major) storage
 *       order, or has non-standard strides. It may or may not work.
 */
template<>
struct type_caster<eos::core::Image3u>
{
	bool load(handle src, bool)
	{
		auto buf = pybind11::array::ensure(src);
		if (!buf)
			return false;

		// Todo: We should probably check that buf.strides(i) is "default", by dividing it by the Scalar type or something.

		if (!pybind11::isinstance<pybind11::array_t<std::uint8_t>>(buf))
		{
			return false; // we only convert uint8_t for now.
		}

		if (buf.ndim() != 3) {
			return false; // we expected a numpy array with 3 dimensions.
		}
		// We got something with 3 dimensions, i.e. an image with 2, 3 or 4 channels (or 'k' for that matter):
		if (buf.shape(2) != 3) {
			return false; // We expected a 3-channel image.
		}
		
		// Note: If our Image class had support for col/row major, we could just map buf.mutable_data().
		// Like with OpenCV: value = cv::Mat(buf.shape(0), buf.shape(1), opencv_type, buf.mutable_data());
		// But since it doesn't, we just copy the data for now:
		value = eos::core::Image3u(buf.shape(0), buf.shape(1));
		array_t<std::uint8_t> buf_as_array(buf);
		for (int r = 0; r < buf.shape(0); ++r) {
			for (int c = 0; c < buf.shape(1); ++c) {
				value(r, c)[0] = buf_as_array.at(r, c, 0);
				value(r, c)[1] = buf_as_array.at(r, c, 1);
				value(r, c)[2] = buf_as_array.at(r, c, 2);
			}
		}
		return true;
	};

    static handle cast(const eos::core::Image3u& src, return_value_policy /* policy */, handle /* parent */)
	{
		const std::size_t num_channels = 3;
		std::vector<std::size_t> shape = { static_cast<std::size_t>(src.height()), static_cast<std::size_t>(src.width()), num_channels };

		// (2048, 4, 1) is default which results in transposed image
		// Below line works now. In numpy the strides are (2048, 4, 1) though. I think a copy gets created nevertheless?
		//std::vector<std::size_t> strides = { num_channels, num_channels * src.height(), 1 }; // might be cols or rows...? I think rows?
		std::vector<std::size_t> strides = { num_channels * src.width(), num_channels, 1 }; // This seems to work with the row-major Image class. I just returned the same strides as the ones we got from NumPy.
		// Note: I think with the change to the new Image class (July 2018), which is now row-major by default, the strides here might have changed. I didn't check.
		// Also, since the new Image stores a vector of Pixels, the question is whether there's any padding added by the compiler, but probably not, since all types that we use are 1 byte or a multiple thereof.
		// numpy: 'f' = fortran = col-major
		//        'c' = c = row-major = default I think.
		return array(pybind11::dtype::of<std::uint8_t>(), shape, strides, &src(0, 0).data()[0]).release();
	};

    PYBIND11_TYPE_CASTER(eos::core::Image3u, _("numpy.ndarray[uint8[m, n, 3]]"));
};

/**
 * @brief Transparent conversion for eos::core::Image4u to and from Python.
 *
 * Converts an eos::core::Image4u to and from Python. Can construct a eos::core::Image4u from numpy arrays,
 * as well as potentially other Python array types.
 *
 * Note: Not sure what happens if the given numpy array is not contiguous, not in default (row-major) storage
 *       order, or has non-standard strides. It may or may not work.
 */
template<>
struct type_caster<eos::core::Image4u>
{
	bool load(handle src, bool)
	{
		auto buf = pybind11::array::ensure(src);
		if (!buf)
			return false;

		// Todo: We should probably check that buf.strides(i) is "default", by dividing it by the Scalar type or something.

		if (!pybind11::isinstance<pybind11::array_t<std::uint8_t>>(buf))
		{
			return false; // we only convert uint8_t for now.
		}

		if (buf.ndim() != 3) {
			return false; // we expected a numpy array with 3 dimensions.
		}
		// We got something with 3 dimensions, i.e. an image with 2, 3 or 4 channels (or 'k' for that matter):
		if (buf.shape(2) != 4) {
			return false; // We expected a 4-channel image.
		}

		// Note: If our Image class had support for col/row major, we could just map buf.mutable_data().
		// Like with OpenCV: value = cv::Mat(buf.shape(0), buf.shape(1), opencv_type, buf.mutable_data());
		// But since it doesn't, we just copy the data for now:
		value = eos::core::Image4u(buf.shape(0), buf.shape(1));
		array_t<std::uint8_t> buf_as_array(buf);
		for (int r = 0; r < buf.shape(0); ++r) {
			for (int c = 0; c < buf.shape(1); ++c) {
				value(r, c)[0] = buf_as_array.at(r, c, 0);
				value(r, c)[1] = buf_as_array.at(r, c, 1);
				value(r, c)[2] = buf_as_array.at(r, c, 2);
				value(r, c)[3] = buf_as_array.at(r, c, 3);
			}
		}
		return true;
	};

    static handle cast(const eos::core::Image4u& src, return_value_policy /* policy */, handle /* parent */)
	{
		const std::size_t num_channels = 4;
		std::vector<std::size_t> shape;
		shape = { static_cast<std::size_t>(src.height()), static_cast<std::size_t>(src.width()), num_channels };

		// (2048, 4, 1) is default which results in transposed image
		// Below line works now. In numpy the strides are (2048, 4, 1) though. I think a copy gets created nevertheless?
		std::vector<size_t> strides = { num_channels * src.width(), num_channels, 1 }; // might be cols or rows...? I think rows?
		// Note: I think with the change to the new Image class (July 2018), which is now row-major by default, the strides here might have changed. I didn't check.
		// Also, since the new Image stores a vector of Pixels, the question is whether there's any padding added by the compiler, but probably not, since all types that we use are 1 byte or a multiple thereof.
		return array(pybind11::dtype::of<std::uint8_t>(), shape, strides, &src(0, 0).data()[0]).release();
	};

    PYBIND11_TYPE_CASTER(eos::core::Image4u, _("numpy.ndarray[uint8[m, n, 4]]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#endif /* EOS_PYBIND11_IMAGE_HPP_ */
