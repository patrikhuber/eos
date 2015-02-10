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
	MorphableModel(PcaModel shapeModel, PcaModel colorModel, std::vector<cv::Vec2f> textureCoordinates = std::vector<cv::Vec2f>());

	/**
	 * Returns the PCA shape model of this Morphable Model.
	 * as a Mesh.
	 *
	 * @return The shape model.
	 */
	PcaModel getShapeModel() const;
	
	/**
	 * Returns the PCA color or albedo model of this Morphable Model.
	 *
	 * @return The color model.
	 */	
	PcaModel getColorModel() const;

	/**
	 * Returns the mean of the shape- and color model as a Mesh.
	 *
	 * @return An mesh instance of the mean of the Morphable Model.
	 */
	render::Mesh getMean() const;

	/**
	 * Draws a random sample from the model, where the coefficients
	 * for the shape- and color models are both drawn from a standard
	 * normal (or with the given standard deviation).
	 *
	 * @param[in] shapeSigma The shape model standard deviation.
	 * @param[in] colorSigma The color model standard deviation.
	 * @return A random sample from the model.
	 */
	render::Mesh drawSample(float shapeSigma = 1.0f, float colorSigma = 1.0f);

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
	render::Mesh drawSample(std::vector<float> shapeCoefficients, std::vector<float> colorCoefficients);

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

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* MORPHABLEMODEL_HPP_ */
