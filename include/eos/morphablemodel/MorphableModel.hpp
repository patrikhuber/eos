/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/MorphableModel.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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

#ifndef MORPHABLEMODEL_HPP_
#define MORPHABLEMODEL_HPP_

#include "eos/morphablemodel/PcaModel.hpp"

#include "eos/render/Mesh.hpp"

#include "eos/morphablemodel/io/mat_cerealisation.hpp"
#include "cereal/access.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"

#include <vector>
#include <array>

// Forward declaration of an internal function
namespace eos { namespace morphablemodel { namespace detail {
	eos::render::Mesh sample_to_mesh(cv::Mat shape, cv::Mat color, std::vector<std::array<int, 3>> tvi, std::vector<std::array<int, 3>> tci, std::vector<cv::Vec2f> texture_coordinates = std::vector<cv::Vec2f>());
} } }

namespace eos {
	namespace morphablemodel {

/**
 * @brief A class representing a 3D Morphable Model, consisting
 * of a shape- and colour (albedo) PCA model.
 *
 * For the general idea of 3DMMs see T. Vetter, V. Blanz,
 * 'A Morphable Model for the Synthesis of 3D Faces', SIGGRAPH 1999.
 */
class MorphableModel
{
public:
	MorphableModel() = default;

	/**
	 * Create a Morphable Model from a shape and a color PCA model, and optional
	 * texture coordinates.
	 *
	 * @param[in] shape_model A PCA model over the shape.
	 * @param[in] color_model A PCA model over the colour (albedo).
	 * @param[in] texture_coordinates Optional texture coordinates for every vertex.
	 */
	MorphableModel(PcaModel shape_model, PcaModel color_model, std::vector<cv::Vec2f> texture_coordinates = std::vector<cv::Vec2f>()) : shape_model(shape_model), color_model(color_model), texture_coordinates(texture_coordinates)
	{
	};

	/**
	 * Returns the PCA shape model of this Morphable Model.
	 * as a Mesh.
	 *
	 * @return The shape model.
	 */
	PcaModel get_shape_model() const
	{
		return shape_model;
	};
	
	/**
	 * Returns the PCA color (albedo) model of this Morphable Model.
	 *
	 * @return The color model.
	 */
	PcaModel get_color_model() const
	{
		return color_model;
	};

	/**
	 * Returns the mean of the shape- and color model as a Mesh.
	 *
	 * @return An mesh instance of the mean of the Morphable Model.
	 */
	render::Mesh get_mean() const
	{
		assert(shape_model.get_data_dimension() == color_model.get_data_dimension() || !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models, or, alternatively, it has to be a shape-only model.

		cv::Mat shape = shape_model.get_mean();
		cv::Mat color = color_model.get_mean();

		render::Mesh mesh;
		if (has_texture_coordinates()) {
			mesh = detail::sample_to_mesh(shape, color, shape_model.get_triangle_list(), color_model.get_triangle_list(), texture_coordinates);
		}
		else {
			mesh = detail::sample_to_mesh(shape, color, shape_model.get_triangle_list(), color_model.get_triangle_list());
		}
		return mesh;
	};

	/**
	 * Draws a random sample from the model, where the coefficients
	 * for the shape- and color models are both drawn from a standard
	 * normal (or with the given standard deviation).
	 *
	 * @param[in] shape_sigma The shape model standard deviation.
	 * @param[in] color_sigma The color model standard deviation.
	 * @return A random sample from the model.
	 */
	render::Mesh draw_sample(float shape_sigma = 1.0f, float color_sigma = 1.0f)
	{
		assert(shape_model.get_data_dimension() == color_model.get_data_dimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

		cv::Mat shapeSample = shape_model.draw_sample(shape_sigma);
		cv::Mat colorSample = color_model.draw_sample(color_sigma);

		render::Mesh mesh;
		if (has_texture_coordinates()) {
			mesh = detail::sample_to_mesh(shapeSample, colorSample, shape_model.get_triangle_list(), color_model.get_triangle_list(), texture_coordinates);
		}
		else {
			mesh = detail::sample_to_mesh(shapeSample, colorSample, shape_model.get_triangle_list(), color_model.get_triangle_list());
		}
		return mesh;
	};

	/**
	 * Returns a sample from the model with the given shape- and
	 * colour PCA coefficients.
	 *
	 * If one of the given vectors is empty, the mean is used.
	 * The coefficient vectors should contain normalised, i.e. standard normal distributed coefficients.
	 * If the Morphable Model is a shape-only model (without colour model), make sure to
	 * leave \c color_coefficients empty.
	 *
	 * @param[in] shape_coefficients The PCA coefficients used to generate the shape sample.
	 * @param[in] color_coefficients The PCA coefficients used to generate the vertex colouring.
	 * @return A model instance with given coefficients.
	 */
	render::Mesh draw_sample(std::vector<float> shape_coefficients, std::vector<float> color_coefficients)
	{
		assert(shape_model.get_data_dimension() == color_model.get_data_dimension() || !has_color_model()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models, or, alternatively, it has to be a shape-only model.

		cv::Mat shape_sample;
		cv::Mat color_sample;

		if (shape_coefficients.empty()) {
			shape_sample = shape_model.get_mean();
		}
		else {
			shape_sample = shape_model.draw_sample(shape_coefficients);
		}
		if (color_coefficients.empty()) {
			color_sample = color_model.get_mean();
		}
		else {
			color_sample = color_model.draw_sample(color_coefficients);
		}

		render::Mesh mesh;
		if (has_texture_coordinates()) {
			mesh = detail::sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(), color_model.get_triangle_list(), texture_coordinates);
		}
		else {
			mesh = detail::sample_to_mesh(shape_sample, color_sample, shape_model.get_triangle_list(), color_model.get_triangle_list());
		}
		return mesh;
	};

	/**
	 * Returns true if this Morphable Model contains a colour
	 * model. Returns false if it is a shape-only model.
	 *
	 * @return True if the Morphable Model has a colour model (i.e. is not a shape-only model).
	 */
	bool has_color_model() const
	{
		return !color_model.get_mean().empty();
	};

private:
	PcaModel shape_model; ///< A PCA model of the shape
	PcaModel color_model; ///< A PCA model of vertex color information
	std::vector<cv::Vec2f> texture_coordinates; ///< uv-coordinates for every vertex

	/**
	 * Returns whether the model has texture mapping coordinates, i.e.
	 * coordinates for every vertex.
	 *
	 * @return True if the model contains texture mapping coordinates.
	 */
	bool has_texture_coordinates() const {
		return texture_coordinates.size() > 0 ? true : false;
	};

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(shape_model, color_model, texture_coordinates);
	};
};

/**
 * Helper method to load a Morphable Model from
 * a cereal::BinaryInputArchive from the harddisk.
 *
 * @param[in] filename Filename to a model.
 * @return The loaded Morphable Model.
 * @throw std::runtime_error When the file given in \c filename fails to be opened (most likely because the file doesn't exist).
 */
MorphableModel load_model(std::string filename)
{
	MorphableModel model;

	std::ifstream file(filename, std::ios::binary);
	if (file.fail()) {
		throw std::runtime_error("Error opening given file: " + filename);
	}
	cereal::BinaryInputArchive input_archive(file);
	input_archive(model);

	return model;
};

/**
 * Helper method to save a Morphable Model to the
 * harddrive as cereal::BinaryInputArchive.
 *
 * @param[in] model The model to be saved.
 * @param[in] filename Filename for the model.
 */
void save_model(MorphableModel model, std::string filename)
{
	std::ofstream file(filename, std::ios::binary);
	cereal::BinaryOutputArchive output_archive(file);
	output_archive(model);
};


namespace detail { /* eos::morphablemodel::detail */
/**
 * Internal helper function that creates a Mesh from given shape and colour
 * PCA instances. Needs the vertex index lists as well to assemble the mesh -
 * and optional texture coordinates.
 *
 * If \c color is empty, it will create a mesh without vertex colouring.
 *
 * @param[in] shape PCA shape model instance.
 * @param[in] color PCA color model instance.
 * @param[in] tvi Triangle vertex indices.
 * @param[in] tci Triangle color indices (usually identical to the vertex indices).
 * @param[in] texture_coordinates Optional texture coordinates for each vertex.
 * @return A mesh created from given parameters.
 */
eos::render::Mesh sample_to_mesh(cv::Mat shape, cv::Mat color, std::vector<std::array<int, 3>> tvi, std::vector<std::array<int, 3>> tci, std::vector<cv::Vec2f> texture_coordinates /* = std::vector<cv::Vec2f>() */)
{
	assert(shape.rows == color.rows || color.empty()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models, or, alternatively, it has to be a shape-only model.

	auto num_vertices = shape.rows / 3;

	eos::render::Mesh mesh;

	// Construct the mesh vertices:
	mesh.vertices.resize(num_vertices);
	for (auto i = 0; i < num_vertices; ++i) {
		mesh.vertices[i] = cv::Vec4f(shape.at<float>(i * 3 + 0), shape.at<float>(i * 3 + 1), shape.at<float>(i * 3 + 2), 1.0f);
	}

	// Assign the vertex color information if it's not a shape-only model:
	if (!color.empty()) {
		mesh.colors.resize(num_vertices);
		for (auto i = 0; i < num_vertices; ++i) {
			mesh.colors[i] = cv::Vec3f(color.at<float>(i * 3 + 0), color.at<float>(i * 3 + 1), color.at<float>(i * 3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
		}
	}

	// Assign the triangle lists:
	mesh.tvi = tvi;
	mesh.tci = tci; // tci will be empty in case of a shape-only model

	// Texture coordinates, if the model has them:
	if (!texture_coordinates.empty()) {
		mesh.texcoords.resize(num_vertices);
		for (auto i = 0; i < num_vertices; ++i) {
			mesh.texcoords[i] = texture_coordinates[i];
		}
	}

	return mesh;
};
} /* namespace eos::morphablemodel::detail */

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* MORPHABLEMODEL_HPP_ */
