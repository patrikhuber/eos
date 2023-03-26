/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/io/eigen_cerealisation.hpp
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

#ifndef EOS_EIGEN_MATRIX_BINARY_CEREALISATION_HPP
#define EOS_EIGEN_MATRIX_BINARY_CEREALISATION_HPP

#include "cereal/cereal.hpp"

#include "Eigen/Core"

#include <cstdint>

/**
 * @brief Serialisation of Eigen matrices for the serialisation
 * library cereal (http://uscilab.github.io/cereal/index.html).
 *
 * Contains serialisation for Eigen matrices to binary archives, i.e. matrices like
 * \c Eigen::MatrixXf, \c Eigen::Matrix4d, or \c Eigen::Vector3f.
 *
 * Todo: Add serialisation to and from JSON. Need to find out how to combine the two
 * variants of SFINAE that are used.
 */
namespace cereal {

/**
 * @brief Serialise an Eigen::Matrix using cereal.
 *
 * Note: Writes the binary data from Matrix::data(), so not sure what happens if a matrix ever has
 * non-contiguous data (if that can ever happen with Eigen).
 *
 * @param[in] ar The archive to serialise to.
 * @param[in] matrix The matrix to serialise.
 */
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline
typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
save(Archive& ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
{
    const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
    const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
    ar(rows);
    ar(cols);
    ar(binary_data(matrix.data(), rows * cols * sizeof(_Scalar)));
};

/**
 * @brief De-serialise an Eigen::Matrix using cereal.
 *
 * Reads the block of binary data back from a cereal archive into the Eigen::Matrix.
 *
 * @param[in] ar The archive to deserialise from.
 * @param[in] matrix The matrix to deserialise into.
 */
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
inline
typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
{
    std::int32_t rows;
    std::int32_t cols;
    ar(rows);
    ar(cols);

    matrix.resize(rows, cols);

    ar(binary_data(matrix.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
};

} /* namespace cereal */

#endif /* EOS_EIGEN_MATRIX_BINARY_CEREALISATION_HPP */
