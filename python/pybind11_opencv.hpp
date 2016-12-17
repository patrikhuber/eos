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

#include <iostream>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/**
 * @file python/pybind11_opencv.hpp
 * @brief Transparent conversion to and from Python for OpenCV vectors.
 */

template<typename T, int N>
struct type_caster<cv::Vec<T, N>>
{
	using vector_type = cv::Vec<T, N>;
	using Scalar = T;
	static constexpr std::size_t num_elements = N;

	bool load(handle src, bool)
	{
		array_t<Scalar> buf(src, true);
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

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
