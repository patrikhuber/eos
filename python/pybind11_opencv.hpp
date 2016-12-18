/*
    python/pybind11_opencv.hpp: Transparent conversion for OpenCV cv::Mat matrices.
                                This header is based on pybind11/eigen.h.

    Copyright (c) 2016 Patrik Huber

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in pybind11's LICENSE file.
*/
#pragma once

#include "pybind11/numpy.h"

#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"

#include <cstddef>
#include <iostream>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/**
 * @file python/pybind11_opencv.hpp
 * @brief Transparent conversion to and from Python for OpenCV vector and matrix types.
 *
 * OpenCV and numpy both use row-major storage order by default, so the conversion works
 * pretty much out of the box, even for multi-channel matrices.
 * Handling of non-standard strides is not implemented.
 * Handling of column-major numpy arrays is unsupported.
 */

/**
 * @brief Transparent conversion for OpenCV's cv::Vec types to and from Python.
 */
template<typename T, int N>
struct type_caster<cv::Vec<T, N>>
{
	using vector_type = cv::Vec<T, N>;
	using Scalar = T;
	static constexpr std::size_t num_elements = N;

	bool load(handle src, bool)
	{
		auto buf = array_t<Scalar>::ensure(src);
		if (!buf)
			return false;

		if (buf.ndim() == 1) // a 1-dimensional vector
		{
			if (buf.shape(0) != num_elements) {
				return false; // not a N-elements vector (can this ever happen?)
			}
			if (buf.strides(0) != sizeof(Scalar))
			{
				std::cout << "An array with non-standard strides is given. Please pass a contiguous array." << std::endl;
				return false;
			}
			value = cv::Vec<T, N>(buf.mutable_data());
		}
		else { // buf.ndim() != 1
			return false;
		}
		return true;
	}

	static handle cast(const vector_type& src, return_value_policy /* policy */, handle /* parent */)
	{
		return array(
			num_elements,	// shape
			src.val			// data
		).release();
	}

	// Specifies the doc-string for the type in Python:
	PYBIND11_TYPE_CASTER(vector_type, _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name() +
		_("[") + _<num_elements>() + _("]]"));
};

/**
 * @brief Helper function to convert a Python array to a cv::Mat.
 *
 * This is an internal helper function that converts a pybind11::array to a cv::Mat.
 * - The \p opencv_depth given must match the type of the data in \p buf.
 * - \p buf must be a 1, 2 or 3-dimensional array.
 * - The function expects a valid \p buf object.
 *
 * The buffer's mutable_data() is used directly, and I think no data is copied.
 *
 * @param buf Python buffer object.
 * @param opencv_depth OpenCV "depth" (the data type, e.g. CV_8U).
 * @return A cv::Mat pointing to the buffer's data or an empty Mat if an error occured.
 */
cv::Mat pyarray_to_mat(pybind11::array buf, int opencv_depth)
{
	cv::Mat value;

	if (buf.ndim() == 1)
	{ // A numpy array, with only one dimension. A row-vector.
		auto num_elements = buf.shape(0);
		auto opencv_type = CV_MAKETYPE(opencv_depth, 1);
		value = cv::Mat(1, num_elements, opencv_type, buf.mutable_data());
	}
	else if (buf.ndim() == 2)
	{ // We got a matrix (but it can also be 1 x n or n x 1)
		auto opencv_type = CV_MAKETYPE(opencv_depth, 1);
		value = cv::Mat(buf.shape(0), buf.shape(1), opencv_type, buf.mutable_data());
	}
	else if (buf.ndim() == 3)
	{ // We got something with 3 dimensions, i.e. an image with 2, 3 or 4 channels (or 'k' for that matter)
		auto num_chans = buf.shape(2); // Check whether 3 or 4 and abort otherwise??
		auto opencv_type = CV_MAKETYPE(opencv_depth, num_chans);
		value = cv::Mat(buf.shape(0), buf.shape(1), opencv_type, buf.mutable_data());
	}
	else { // buf.ndim() is not 1, 2 or 3.
		return cv::Mat();
	}
	return value;
};

/**
 * @brief Transparent conversion for OpenCV's cv::Mat type to and from Python.
 *
 * Converts cv::Mat's to and from Python. Can construct a cv::Mat from numpy arrays,
 * as well as potentially other Python array types.
 *
 * - Supports only contiguous matrices
 * - The numpy array has to be in default (row-major) storage order
 * - Non-default strides are not implemented.
 *
 * Note about strides: http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#mat-step1
 * And possibly use src.elemSize or src.elemSize1.
 * See also the old bindings: https://github.com/patrikhuber/eos/commit/1c3b0113a3efcbb1a92efca646be663ef8593793
 */
template<>
struct type_caster<cv::Mat>
{
    bool load(handle src, bool)
	{
		// Since cv::Mat has its time dynamically specified at run-time, we can't bind functions
		// that take a cv::Mat to any particular Scalar type.
		// Thus the data we get from python can be any type.
		auto buf = pybind11::array::ensure(src);
		if (!buf)
			return false;

		// Todo: We should probably check that buf.strides(i) is "default", by dividing it by the Scalar type or something.

		int opencv_depth;
		if (pybind11::isinstance<pybind11::array_t<std::uint8_t>>(buf))
		{
			opencv_depth = CV_8U;
		}
		else if (pybind11::isinstance<pybind11::array_t<std::int32_t>>(buf))
		{
			opencv_depth = CV_32S;
		}
		else if (pybind11::isinstance<pybind11::array_t<float>>(buf))
		{
			opencv_depth = CV_32F;
		}
		else if (pybind11::isinstance<pybind11::array_t<double>>(buf))
		{
			opencv_depth = CV_64F;
		}
		else
		{
			return false;
		}

		value = pyarray_to_mat(buf, opencv_depth);
		if (value.empty())
		{
			return false;
		}
		return true;
	};

    static handle cast(const cv::Mat& src, return_value_policy /* policy */, handle /* parent */)
	{
		if (!src.isContinuous())
		{
			throw std::runtime_error("Cannot cast non-contiguous cv::Mat objects to Python. Change the C++ code to return a contiguous cv::Mat.");
			// We could probably support that with implementing strides properly.
		}

		const auto opencv_depth = src.depth();
		const auto num_chans = src.channels();
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
		if (opencv_depth == CV_8U)
		{
			return array(pybind11::dtype::of<std::uint8_t>(), shape, src.data).release();
		}
		else if (opencv_depth == CV_32S)
		{
			return array(pybind11::dtype::of<std::int32_t>(), shape, src.data).release();
		}
		else if (opencv_depth == CV_32F)
		{
			return array(pybind11::dtype::of<float>(), shape, src.data).release();
		}
		else if (opencv_depth == CV_64F)
		{
			return array(pybind11::dtype::of<double>(), shape, src.data).release();
		}
		else {
			throw std::runtime_error("Can currently only return matrices of type 8U, 32S, 32F and 64F back to Python. Other types can be added if needed.");
		}
	};

    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray[uint8|int32|float32|float64[m, n, d]]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
