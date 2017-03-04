/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/io/cvssp.hpp
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

#ifndef IO_CVSSP_HPP_
#define IO_CVSSP_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include "boost/filesystem/path.hpp"

#include <vector>
#include <iostream>

namespace eos {
	namespace morphablemodel {

// Forward declaration
std::vector<std::array<double, 2>> load_isomap(boost::filesystem::path isomap_file);

/**
 * Load a shape or color model from a .scm file containing
 * a Morphable Model in the Surrey format. CVSSP's software
 * internally trains and stores the model in this custom binary
 * format and this class provides means to load them.
 *
 * Note on multi-resolution models: The landmarks to vertex-id mapping is
 * always the same. The lowest resolution model has all the landmarks defined
 * and for the higher resolutions, the mesh is divided from that on.
 * Note: For new landmarks we add, this might not be the case if we add them
 * in the highest resolution model, so take care!
 *
 * - The PCA basis matrix stored in the file and loaded is the orthogonal PCA basis, i.e. it is not normalised by the eigenvalues.
 *
 * @param[in] model_filename A binary .scm-file containing the model.
 * @param[in] isomap_file An optional path to an isomap containing texture coordinates.
 * @return The Morphable Model loaded from the file.
 * @throws ...
 */
inline MorphableModel load_scm_model(boost::filesystem::path model_filename, boost::filesystem::path isomap_file = boost::filesystem::path())
{
	using cv::Mat;
	if (sizeof(unsigned int) != 4) { // note/todo: maybe use uint32 or similar instead? Yep, but still we could encounter endianness-trouble.
		std::cout << "Warning: We're reading 4 Bytes from the file but sizeof(unsigned int) != 4. Check the code/behaviour." << std::endl;
	}
	if (sizeof(double) != 8) {
		std::cout << "Warning: We're reading 8 Bytes from the file but sizeof(double) != 8. Check the code/behaviour." << std::endl;
	}

	std::ifstream modelFile(model_filename.string(), std::ios::binary);
	if (!modelFile.is_open()) {
		std::string msg("Unable to open model file: " + model_filename.string());
		std::cout << msg << std::endl;
		throw std::runtime_error(msg);
	}

	// Reading the shape model
	// Read (reference?) num triangles and vertices
	unsigned int numVertices = 0;
	unsigned int numTriangles = 0;
	modelFile.read(reinterpret_cast<char*>(&numVertices), 4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
	modelFile.read(reinterpret_cast<char*>(&numTriangles), 4);

	// Read triangles
	std::vector<std::array<int, 3>> triangleList;

	triangleList.resize(numTriangles);
	unsigned int v0, v1, v2;
	for (unsigned int i = 0; i < numTriangles; ++i) {
		v0 = v1 = v2 = 0;
		modelFile.read(reinterpret_cast<char*>(&v0), 4);	// would be nice to pass a &vector and do it in one
		modelFile.read(reinterpret_cast<char*>(&v1), 4);	// go, but didn't work. Maybe a cv::Mat would work?
		modelFile.read(reinterpret_cast<char*>(&v2), 4);
		triangleList[i][0] = v0;
		triangleList[i][1] = v1;
		triangleList[i][2] = v2;
	}

	// Read number of rows and columns of the shape projection matrix (pcaBasis)
	unsigned int numShapePcaCoeffs = 0;
	unsigned int numShapeDims = 0;	// dimension of the shape vector (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numShapePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numShapeDims), 4);

	if (3 * numVertices != numShapeDims) {
		std::cout << "Warning: Number of shape dimensions is not equal to three times the number of vertices. Something will probably go wrong during the loading." << std::endl;
	}

	// Read shape projection matrix
	Mat unnormalisedPcaBasisShape(numShapeDims, numShapePcaCoeffs, CV_32FC1); // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << "Loading shape PCA basis matrix with " << unnormalisedPcaBasisShape.rows << " rows and " << unnormalisedPcaBasisShape.cols << " cols." << std::endl;
	for (unsigned int col = 0; col < numShapePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numShapeDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			unnormalisedPcaBasisShape.at<float>(row, col) = static_cast<float>(var);
		}
	}

	// Read mean shape vector
	unsigned int numMean = 0; // dimension of the mean (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numMean), 4);
	if (numMean != numShapeDims) {
		std::cout << "Warning: Number of shape dimensions is not equal to the number of dimensions of the mean. Something will probably go wrong during the loading." << std::endl;
	}
	Mat meanShape(numMean, 1, CV_32FC1);
	unsigned int counter = 0;
	double vd0, vd1, vd2;
	for (unsigned int i = 0; i < numMean / 3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8);
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		meanShape.at<float>(counter, 0) = static_cast<float>(vd0);
		++counter;
		meanShape.at<float>(counter, 0) = static_cast<float>(vd1);
		++counter;
		meanShape.at<float>(counter, 0) = static_cast<float>(vd2);
		++counter;
	}

	// Read shape eigenvalues
	unsigned int numEigenValsShape = 0;
	modelFile.read(reinterpret_cast<char*>(&numEigenValsShape), 4);
	if (numEigenValsShape != numShapePcaCoeffs) {
		std::cout << "Warning: Number of coefficients in the PCA basis matrix is not equal to the number of eigenvalues. Something will probably go wrong during the loading." << std::endl;
	}
	Mat eigenvaluesShape(numEigenValsShape, 1, CV_32FC1);
	for (unsigned int i = 0; i < numEigenValsShape; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenvaluesShape.at<float>(i, 0) = static_cast<float>(var);
	}

	// We read the unnormalised basis from the file. Now let's normalise it and store the normalised basis separately.
	// Todo: We should change these to read into an Eigen matrix directly, and not into a cv::Mat first.
	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> unnormalisedPcaBasisShape_(unnormalisedPcaBasisShape.ptr<float>(), unnormalisedPcaBasisShape.rows, unnormalisedPcaBasisShape.cols);
	Eigen::Map<RowMajorMatrixXf> eigenvaluesShape_(eigenvaluesShape.ptr<float>(), eigenvaluesShape.rows, eigenvaluesShape.cols);
	Eigen::Map<RowMajorMatrixXf> meanShape_(meanShape.ptr<float>(), meanShape.rows, meanShape.cols);
	Eigen::MatrixXf normalisedPcaBasisShape_ = rescale_pca_basis(unnormalisedPcaBasisShape_, eigenvaluesShape_);
	PcaModel shapeModel(meanShape_, normalisedPcaBasisShape_, eigenvaluesShape_, triangleList);

	// Reading the color model
	// Read number of rows and columns of projection matrix
	unsigned int numTexturePcaCoeffs = 0;
	unsigned int numTextureDims = 0;
	modelFile.read(reinterpret_cast<char*>(&numTexturePcaCoeffs), 4);
	modelFile.read(reinterpret_cast<char*>(&numTextureDims), 4);
	// Read color projection matrix
	Mat unnormalisedPcaBasisColor(numTextureDims, numTexturePcaCoeffs, CV_32FC1);
	std::cout << "Loading color PCA basis matrix with " << unnormalisedPcaBasisColor.rows << " rows and " << unnormalisedPcaBasisColor.cols << " cols." << std::endl;
	for (unsigned int col = 0; col < numTexturePcaCoeffs; ++col) {
		for (unsigned int row = 0; row < numTextureDims; ++row) {
			double var = 0.0;
			modelFile.read(reinterpret_cast<char*>(&var), 8);
			unnormalisedPcaBasisColor.at<float>(row, col) = static_cast<float>(var);
		}
	}

	// Read mean color vector
	unsigned int numMeanColor = 0; // dimension of the mean (3*numVertices)
	modelFile.read(reinterpret_cast<char*>(&numMeanColor), 4);
	Mat meanColor(numMeanColor, 1, CV_32FC1);
	counter = 0;
	for (unsigned int i = 0; i < numMeanColor / 3; ++i) {
		vd0 = vd1 = vd2 = 0.0;
		modelFile.read(reinterpret_cast<char*>(&vd0), 8); // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
		modelFile.read(reinterpret_cast<char*>(&vd1), 8);
		modelFile.read(reinterpret_cast<char*>(&vd2), 8);
		meanColor.at<float>(counter, 0) = static_cast<float>(vd0);
		++counter;
		meanColor.at<float>(counter, 0) = static_cast<float>(vd1);
		++counter;
		meanColor.at<float>(counter, 0) = static_cast<float>(vd2);
		++counter;
	}

	// Read color eigenvalues
	unsigned int numEigenValsColor = 0;
	modelFile.read(reinterpret_cast<char*>(&numEigenValsColor), 4);
	Mat eigenvaluesColor(numEigenValsColor, 1, CV_32FC1);
	for (unsigned int i = 0; i < numEigenValsColor; ++i) {
		double var = 0.0;
		modelFile.read(reinterpret_cast<char*>(&var), 8);
		eigenvaluesColor.at<float>(i, 0) = static_cast<float>(var);
	}

	// We read the unnormalised basis from the file. Now let's normalise it and store the normalised basis separately.
	Eigen::Map<RowMajorMatrixXf> unnormalisedPcaBasisColor_(unnormalisedPcaBasisColor.ptr<float>(), unnormalisedPcaBasisColor.rows, unnormalisedPcaBasisColor.cols);
	Eigen::Map<RowMajorMatrixXf> eigenvaluesColor_(eigenvaluesColor.ptr<float>(), eigenvaluesColor.rows, eigenvaluesColor.cols);
	Eigen::Map<RowMajorMatrixXf> meanColor_(meanColor.ptr<float>(), meanColor.rows, meanColor.cols);
	Eigen::MatrixXf normalisedPcaBasisColor_ = rescale_pca_basis(unnormalisedPcaBasisColor_, eigenvaluesColor_);
	PcaModel colorModel(meanColor_, normalisedPcaBasisColor_, eigenvaluesColor_, triangleList);

	modelFile.close();

	// Load the isomap with texture coordinates if a filename has been given:
	std::vector<std::array<double, 2>> texCoords;
	if (!isomap_file.empty()) {
		texCoords = load_isomap(isomap_file);
		if (shapeModel.get_data_dimension() / 3.0f != texCoords.size()) {
			std::string errorMessage("Error, wrong number of texture coordinates. Don't have the same number of texcoords than the shape model has vertices.");
			std::cout << errorMessage << std::endl;
			throw std::runtime_error(errorMessage);
		}
	}

	return MorphableModel(shapeModel, colorModel, texCoords);
};

/**
 * Load a set of 2D texture coordinates pre-generated by the isomap algorithm.
 * After loading, we rescale the coordinates to [0, 1] x [0, 1].
 *
 * @param[in] isomapFile Path to an isomap file containing texture coordinates.
 * @return The 2D texture coordinates for every vertex.
 * @throws ...
 */
inline std::vector<std::array<double, 2>> load_isomap(boost::filesystem::path isomapFile)
{
	using std::string;
	std::vector<float> xCoords, yCoords;
	string line;
	std::ifstream myfile(isomapFile.string());
	if (!myfile.is_open()) {
		string logMessage("The isomap file could not be opened. Did you specify a correct filename? " + isomapFile.string());
		throw std::runtime_error(logMessage);
	}
	else {
		while (getline(myfile, line))
		{
			std::istringstream iss(line);
			string x, y;
			iss >> x >> y;
			xCoords.push_back(std::stof(x));
			yCoords.push_back(std::stof(y));
		}
		myfile.close();
	}
	// Process the coordinates: Find the min/max and rescale to [0, 1] x [0, 1]
	auto minMaxX = std::minmax_element(begin(xCoords), end(xCoords)); // minMaxX is a pair, first=min, second=max
	auto minMaxY = std::minmax_element(begin(yCoords), end(yCoords));

	std::vector<std::array<double, 2>> texCoords;
	float divisorX = *minMaxX.second - *minMaxX.first;
	float divisorY = *minMaxY.second - *minMaxY.first;
	for (int i = 0; i < xCoords.size(); ++i) {
		texCoords.push_back(std::array<double, 2>{(xCoords[i] - *minMaxX.first) / divisorX, 1.0f - (yCoords[i] - *minMaxY.first) / divisorY}); // We rescale to [0, 1] and at the same time flip the y-coords (because in the isomap, the coordinates are stored upside-down).
	}

	return texCoords;
};

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* IO_CVSSP_HPP_ */
