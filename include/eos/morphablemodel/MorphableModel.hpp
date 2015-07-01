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

#include <vector>
#include <array>

/**
 * Forward declaration of an internal function
 */
namespace eos { namespace morphablemodel { namespace detail {
	eos::render::Mesh sampleToMesh(cv::Mat shape, cv::Mat color, std::vector<std::array<int, 3>> tvi, std::vector<std::array<int, 3>> tci, std::vector<cv::Vec2f> textureCoordinates = std::vector<cv::Vec2f>());
} } }

namespace eos {
	namespace morphablemodel {

/**
 * A class representing a 3D Morphable Model.
 * It consists of a shape- and albedo (texture) PCA model.
 * 
 * For the general idea of 3DMMs see T. Vetter, V. Blanz,
 * 'A Morphable Model for the Synthesis of 3D Faces', SIGGRAPH 1999
 */
class MorphableModel
{
public:

	/**
	 * Create a Morphable Model from a shape and a color PCA model, and optional
	 * texture coordinates.
	 *
	 * @param[in] shapeModel A PCA model over the shape.
	 * @param[in] colorModel A PCA model over the color (albedo).
	 * @param[in] textureCoordinates Optional texture coordinates for every vertex.
	 */
	MorphableModel(PcaModel shapeModel, PcaModel colorModel, std::vector<cv::Vec2f> textureCoordinates = std::vector<cv::Vec2f>()) : shapeModel(shapeModel), colorModel(colorModel), textureCoordinates(textureCoordinates)
	{
	};

	/**
	 * Returns the PCA shape model of this Morphable Model.
	 * as a Mesh.
	 *
	 * @return The shape model.
	 */
	PcaModel getShapeModel() const
	{
		return shapeModel;
	};
	
	/**
	 * Returns the PCA color (albedo) model of this Morphable Model.
	 *
	 * @return The color model.
	 */	
	PcaModel getColorModel() const
	{
		return colorModel;
	};

	/**
	 * Returns the mean of the shape- and color model as a Mesh.
	 *
	 * @return An mesh instance of the mean of the Morphable Model.
	 */
	render::Mesh getMean() const
	{
		assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

		cv::Mat shape = shapeModel.getMean();
		cv::Mat color = colorModel.getMean();

		render::Mesh mesh;
		if (hasTextureCoordinates()) {
			mesh = detail::sampleToMesh(shape, color, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
		}
		else {
			mesh = detail::sampleToMesh(shape, color, shapeModel.getTriangleList(), colorModel.getTriangleList());
		}
		return mesh;
	};

	/**
	 * Draws a random sample from the model, where the coefficients
	 * for the shape- and color models are both drawn from a standard
	 * normal (or with the given standard deviation).
	 *
	 * @param[in] shapeSigma The shape model standard deviation.
	 * @param[in] colorSigma The color model standard deviation.
	 * @return A random sample from the model.
	 */
	render::Mesh drawSample(float shapeSigma = 1.0f, float colorSigma = 1.0f)
	{
		assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

		cv::Mat shapeSample = shapeModel.drawSample(shapeSigma);
		cv::Mat colorSample = colorModel.drawSample(colorSigma);

		render::Mesh mesh;
		if (hasTextureCoordinates()) {
			mesh = detail::sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
		}
		else {
			mesh = detail::sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList());
		}
		return mesh;
	};

	/**
	 * Returns a sample from the model with the given shape- and
	 * color PCA coefficients. 
	 *
	 * If one of the given vectors is empty, the mean is used.
	 * The coefficient vectors should contain normalised, i.e. standard normal distributed coefficients.
	 *
	 * @param[in] shapeCoefficients The PCA coefficients used to generate the shape sample.
	 * @param[in] colorCoefficients The PCA coefficients used to generate the shape sample.
	 * @return A model instance with given coefficients.
	 */
	render::Mesh drawSample(std::vector<float> shapeCoefficients, std::vector<float> colorCoefficients)
	{
		assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

		cv::Mat shapeSample;
		cv::Mat colorSample;

		if (shapeCoefficients.empty()) {
			shapeSample = shapeModel.getMean();
		}
		else {
			shapeSample = shapeModel.drawSample(shapeCoefficients);
		}
		if (colorCoefficients.empty()) {
			colorSample = colorModel.getMean();
		}
		else {
			colorSample = colorModel.drawSample(colorCoefficients);
		}

		render::Mesh mesh;
		if (hasTextureCoordinates()) {
			mesh = detail::sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
		}
		else {
			mesh = detail::sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList());
		}
		return mesh;
	};

private:
	PcaModel shapeModel; ///< A PCA model of the shape
	PcaModel colorModel; ///< A PCA model of vertex color information
	std::vector<cv::Vec2f> textureCoordinates; ///< uv-coordinates for every vertex

	/**
	 * Returns whether the model has texture mapping coordinates, i.e.
	 * coordinates for every vertex.
	 *
	 * @return True if the model contains texture mapping coordinates.
	 */
	bool hasTextureCoordinates() const {
		return textureCoordinates.size() > 0 ? true : false;
	};

};

		namespace detail { /* eos::morphablemodel::detail */
/**
 * Internal helper function that creates a Mesh from given shape and color
 * PCA instances. Needs the vertex index lists as well to assemble the mesh -
 * and optional texture coordinates.
 *
 * @param[in] shape PCA shape model instance.
 * @param[in] color PCA color model instance.
 * @param[in] tvi Triangle vertex indices.
 * @param[in] tci Triangle color indices (usually identical to the vertex indices).
 * @param[in] textureCoordinates Optional texture coordinates for each vertex.
 * @return A mesh created from given parameters.
 */
eos::render::Mesh sampleToMesh(cv::Mat shape, cv::Mat color, std::vector<std::array<int, 3>> tvi, std::vector<std::array<int, 3>> tci, std::vector<cv::Vec2f> textureCoordinates /* = std::vector<cv::Vec2f>() */)
{
	assert(shape.rows == color.rows); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.
	
	auto numVertices = shape.rows / 3;
	
	eos::render::Mesh mesh;

	// Construct the mesh vertices and vertex color information:
	mesh.vertices.resize(numVertices);
	mesh.colors.resize(numVertices);
	for (auto i = 0; i < numVertices; ++i) {
		mesh.vertices[i] = cv::Vec4f(shape.at<float>(i * 3 + 0), shape.at<float>(i * 3 + 1), shape.at<float>(i * 3 + 2), 1.0f);
		mesh.colors[i] = cv::Vec3f(color.at<float>(i * 3 + 0), color.at<float>(i * 3 + 1), color.at<float>(i * 3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
	}

	// Assign the triangle lists:
	mesh.tvi = tvi;
	mesh.tci = tci;

	// Texture coordinates, if the model has them:
	if (!textureCoordinates.empty()) {
		mesh.texcoords.resize(numVertices);
		for (auto i = 0; i < numVertices; ++i) {
			mesh.texcoords[i] = textureCoordinates[i];
		}
	}

	return mesh;
};
		} /* namespace detail */

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* MORPHABLEMODEL_HPP_ */
