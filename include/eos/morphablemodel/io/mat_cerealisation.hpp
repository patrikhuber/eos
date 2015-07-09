/*
 * superviseddescent: A C++11 implementation of the supervised descent
 *                    optimisation method
 * File: superviseddescent/matcerealisation.hpp
 *
 * Copyright 2015 Patrik Huber
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

#ifndef MATCEREALISATION_HPP_
#define MATCEREALISATION_HPP_

#include "cereal/cereal.hpp"

#include "opencv2/core/core.hpp"

/**
 * Serialisation for OpenCV cv::Mat matrices for the serialisation
 * library cereal (http://uscilab.github.io/cereal/index.html).
 */

namespace cv {

/**
 * Serialise a cv::Mat using cereal.
 *
 * Supports all types of matrices as well as non-contiguous ones.
 *
 * @param[in] ar The archive to serialise to.
 * @param[in] mat The matrix to serialise.
 */
template<class Archive>
void save(Archive& ar, const cv::Mat& mat)
{
	int rows, cols, type;
	bool continuous;

	rows = mat.rows;
	cols = mat.cols;
	type = mat.type();
	continuous = mat.isContinuous();

	ar & rows & cols & type & continuous;

	if (continuous) {
		const int data_size = rows * cols * static_cast<int>(mat.elemSize());
		auto mat_data = cereal::binary_data(mat.ptr(), data_size);
		ar & mat_data;
	}
	else {
		const int row_size = cols * static_cast<int>(mat.elemSize());
		for (int i = 0; i < rows; i++) {
			auto row_data = cereal::binary_data(mat.ptr(i), row_size);
			ar & row_data;
		}
	}
};

/**
 * De-serialise a cv::Mat using cereal.
 *
 * Supports all types of matrices as well as non-contiguous ones.
 *
 * @param[in] ar The archive to deserialise from.
 * @param[in] mat The matrix to deserialise into.
 */
template<class Archive>
void load(Archive& ar, cv::Mat& mat)
{
	int rows, cols, type;
	bool continuous;

	ar & rows & cols & type & continuous;

	if (continuous) {
		mat.create(rows, cols, type);
		const int data_size = rows * cols * static_cast<int>(mat.elemSize());
		auto mat_data = cereal::binary_data(mat.ptr(), data_size);
		ar & mat_data;
	}
	else {
		mat.create(rows, cols, type);
		const int row_size = cols * static_cast<int>(mat.elemSize());
		for (int i = 0; i < rows; i++) {
			auto row_data = cereal::binary_data(mat.ptr(i), row_size);
			ar & row_data;
		}
	}
};

/**
 * Serialisation for a cv::Vec2f using cereal.
 *
 * @param[in] ar The archive to (de)serialise.
 * @param[in] mat The vector to (de)serialise.
 */
template <class Archive>
void serialize(Archive& ar, cv::Vec2f& vec)
{
	ar(vec[0], vec[1]);
};

}

#endif /* MATCEREALISATION_HPP_ */
