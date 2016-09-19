/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/generate-python-bindings.cpp
 *
 * Copyright 2016 Patrik Huber
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
#include "eos/morphablemodel/PcaModel.hpp"

#include "opencv2/core/core.hpp"

#include "pybind11/pybind11.h"

namespace py = pybind11;

/**
 * Generate python bindings for the eos library using pybind11.
 */
PYBIND11_PLUGIN(eos) {
    py::module m("eos", "Python bindings to the 3D Morphable Face Model fitting library");
	
	py::class_<cv::Vec4f>(m, "Vec4f", "Wrapper for OpenCV's cv::Vec4f type")
		.def_buffer([](cv::Vec4f &vec) -> py::buffer_info {
		return py::buffer_info(
			&vec.val,                               /* Pointer to buffer */
			sizeof(float),                          /* Size of one scalar */
			py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
			2,                                      /* Number of dimensions */
			{ vec.rows, vec.cols },                 /* Buffer dimensions */
			{ sizeof(float),             /* Strides (in bytes) for each index */
			sizeof(float) }					/* => both sizeof(float), since the data is hold in an array, i.e. contiguous memory */
		);
	});

	py::class_<eos::morphablemodel::PcaModel>(m, "PcaModel", "Class representing a PcaModel with a mean, eigenvectors and eigenvalues.")
		.def(py::init<>())
		.def("get_mean_at_point", &eos::morphablemodel::PcaModel::get_mean_at_point, "Returns the mean at the given vertex id.")
		;

    return m.ptr();
};
