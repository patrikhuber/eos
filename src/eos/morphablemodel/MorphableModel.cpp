/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/MorphableModel.cpp
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
#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <iostream>

using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::array;

namespace {
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
eos::render::Mesh sampleToMesh(Mat shape, Mat color, vector<array<int, 3>> tvi, vector<array<int, 3>> tci, vector<Vec2f> textureCoordinates = vector<Vec2f>())
{
	assert(shape.rows == color.rows); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.
	
	auto numVertices = shape.rows / 3;
	
	eos::render::Mesh mesh;

	// Construct the mesh vertices and vertex color information:
	mesh.vertices.resize(numVertices);
	mesh.colors.resize(numVertices);
	for (auto i = 0; i < numVertices; ++i) {
		mesh.vertices[i] = Vec4f(shape.at<float>(i * 3 + 0), shape.at<float>(i * 3 + 1), shape.at<float>(i * 3 + 2), 1.0f);
		mesh.colors[i] = Vec3f(color.at<float>(i * 3 + 0), color.at<float>(i * 3 + 1), color.at<float>(i * 3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
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
}
} /* unnamed namespace */

namespace eos {
	namespace morphablemodel {

MorphableModel::MorphableModel(PcaModel shapeModel, PcaModel colorModel, std::vector<cv::Vec2f> textureCoordinates /*= std::vector<cv::Vec2f>()*/) : shapeModel(shapeModel), colorModel(colorModel), textureCoordinates(textureCoordinates)
{
}


PcaModel MorphableModel::getShapeModel() const
{
	return shapeModel;
}

PcaModel MorphableModel::getColorModel() const
{
	return colorModel;
}

render::Mesh MorphableModel::getMean() const
{
	assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

	Mat shape = shapeModel.getMean();
	Mat color = colorModel.getMean();

	render::Mesh mesh;
	if (hasTextureCoordinates()) {
		mesh = sampleToMesh(shape, color, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
	}
	else {
		mesh = sampleToMesh(shape, color, shapeModel.getTriangleList(), colorModel.getTriangleList());
	}
	return mesh;
}

render::Mesh MorphableModel::drawSample(float shapeSigma /*= 1.0f*/, float colorSigma /*= 1.0f*/)
{
	assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

	Mat shapeSample = shapeModel.drawSample(shapeSigma);
	Mat colorSample = colorModel.drawSample(colorSigma);

	render::Mesh mesh;
	if (hasTextureCoordinates()) {
		mesh = sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
	}
	else {
		mesh = sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList());
	}
	return mesh;
}

render::Mesh MorphableModel::drawSample(vector<float> shapeCoefficients, vector<float> colorCoefficients)
{
	assert(shapeModel.getDataDimension() == colorModel.getDataDimension()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models.

	Mat shapeSample;
	Mat colorSample;

	if (shapeCoefficients.empty()) {
		shapeSample = shapeModel.getMean();
	} else {
		shapeSample = shapeModel.drawSample(shapeCoefficients);
	}
	if (colorCoefficients.empty()) {
		colorSample = colorModel.getMean();
	} else {
		colorSample = colorModel.drawSample(colorCoefficients);
	}

	render::Mesh mesh;
	if (hasTextureCoordinates()) {
		mesh = sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList(), textureCoordinates);
	}
	else {
		mesh = sampleToMesh(shapeSample, colorSample, shapeModel.getTriangleList(), colorModel.getTriangleList());
	}
	return mesh;
}

	} /* namespace morphablemodel */
} /* namespace eos */
