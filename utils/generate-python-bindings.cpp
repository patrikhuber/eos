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
#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace eos;

/**
 * Generate python bindings for the eos library using pybind11.
 */
PYBIND11_PLUGIN(eos) {
    py::module eos_module("eos", "Python bindings to the 3D Morphable Face Model fitting library");
	
	/**
	 * General bindings, for OpenCV vector types and cv::Mat:
	 */
	py::class_<cv::Vec4f>(eos_module, "Vec4f", "Wrapper for OpenCV's cv::Vec4f type")
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

	py::class_<cv::Mat>(eos_module, "Mat")
		.def_buffer([](cv::Mat &mat) -> py::buffer_info {

		if (!mat.isContinuous())
		{
			// I think these throw messages are not shown in Python, it just crashes. Thus, use cout for now.
			std::string error_msg("Only continuous (contiguous) cv::Mat objects are currently supported.");
			std::cout << error_msg << std::endl;
			throw std::runtime_error(error_msg);
		}
		// Note: Also stride/step should be 1 too, but I think this is covered by isContinuous().
		auto dimensions = mat.dims;
		if (dimensions != 2)
		{
			std::string error_msg("Only cv::Mat objects with dims == 2 are currently supported.");
			std::cout << error_msg << std::endl;
			throw std::runtime_error(error_msg);
		}
		if (mat.channels() != 1)
		{
			std::string error_msg("Only cv::Mat objects with channels() == 1 are currently supported.");
			std::cout << error_msg << std::endl;
			throw std::runtime_error(error_msg);
		}

		std::size_t rows = mat.rows;
		std::size_t cols = mat.cols;

		if (mat.type() == CV_32F) {
			return py::buffer_info(
				mat.data,                               /* Pointer to buffer */
				sizeof(float),                          /* Size of one scalar */
				py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
				dimensions,                                      /* Number of dimensions */
				{ rows, cols },                 /* Buffer dimensions */
				{ sizeof(float) * cols,             /* Strides (in bytes) for each index */
				sizeof(float) }	// this way is correct for row-major memory layout (OpenCV)
			);
		}
		else {
			std::string error_msg("Only the cv::Mat type CV_32F is currently supported. If needed, it would be easy to add CV_8U and CV_64F.");
			std::cout << error_msg << std::endl;
			throw std::runtime_error(error_msg);
		}
		// Will never reach here.
	});


	/**
	 * Bindings for the eos::morphablemodel namespace:
	 */
	py::module morphablemodel_module = eos_module.def_submodule("morphablemodel", "Doc for submodule.");

	py::class_<morphablemodel::PcaModel>(morphablemodel_module, "PcaModel", "Class representing a PcaModel with a mean, eigenvectors and eigenvalues, as well as a list of triangles to build a mesh.")
		.def("get_num_principal_components", &morphablemodel::PcaModel::get_num_principal_components, "Returns the number of principal components in the model.")
		.def("get_data_dimension", &morphablemodel::PcaModel::get_data_dimension, "Returns the dimension of the data, i.e. the number of shape dimensions.")
		.def("get_triangle_list", &morphablemodel::PcaModel::get_triangle_list, "Returns a list of triangles on how to assemble the vertices into a mesh.")
		.def("get_mean", &morphablemodel::PcaModel::get_mean, "Returns the mean of the model.")
		.def("get_mean_at_point", &morphablemodel::PcaModel::get_mean_at_point, "Return the value of the mean at a given vertex index.")
		.def("draw_sample", (cv::Mat (morphablemodel::PcaModel::*)(std::vector<float>) const)&morphablemodel::PcaModel::draw_sample, "Returns a sample from the model with the given PCA coefficients. The given coefficients should follow a standard normal distribution, i.e. not be \"normalised\" with their eigenvalues/variances.")
		;

	py::class_<morphablemodel::MorphableModel>(morphablemodel_module, "MorphableModel", "A class representing a 3D Morphable Model, consisting of a shape- and colour (albedo) PCA model, as well as texture (uv) coordinates.")
		.def("get_shape_model", [](const morphablemodel::MorphableModel& m) { return m.get_shape_model(); }, "Returns the PCA shape model of this Morphable Model.") // Not sure if that'll really be const in Python? I think Python does a copy each time this gets called?
		.def("get_color_model", [](const morphablemodel::MorphableModel& m) { return m.get_color_model(); }, "Returns the PCA colour (albedo) model of this Morphable Model.")
		;

	morphablemodel_module.def("load_model", &morphablemodel::load_model, "Load a Morphable Model from a cereal::BinaryInputArchive (.bin) from the harddisk.");

    return eos_module.ptr();
};
