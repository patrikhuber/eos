/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/PcaModel.cpp
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
#include "eos/morphablemodel/PcaModel.hpp"

#include "boost/lexical_cast.hpp"
#include "boost/algorithm/string.hpp"

#include <fstream>

using cv::Mat;
using cv::Vec4f;
using boost::lexical_cast;
using boost::filesystem::path;
using std::string;
using std::vector;
using std::array;

namespace eos {
	namespace morphablemodel {

PcaModel::PcaModel(Mat mean, Mat pcaBasis, Mat eigenvalues, vector<array<int, 3>> triangleList) : mean(mean), normalisedPcaBasis(pcaBasis), eigenvalues(eigenvalues), triangleList(triangleList)
{
	const auto seed = std::random_device()();
	engine.seed(seed);
	unnormalisedPcaBasis = unnormalisePcaBasis(normalisedPcaBasis, eigenvalues);
}

int PcaModel::getNumberOfPrincipalComponents() const
{
	// Note: we could assert(normalisedPcaBasis.cols==unnormalisedPcaBasis.cols)
	return normalisedPcaBasis.cols;
}

int PcaModel::getDataDimension() const
{
	// Note: we could assert(normalisedPcaBasis.rows==unnormalisedPcaBasis.rows)
	return normalisedPcaBasis.rows;
}

std::vector<std::array<int, 3>> PcaModel::getTriangleList() const
{
	return triangleList;
}

Mat PcaModel::getMean() const
{
	return mean;
}

Vec4f PcaModel::getMeanAtPoint(int vertexIndex) const
{
	vertexIndex *= 3;
	if (vertexIndex >= mean.rows) {
		throw std::out_of_range("The given vertex id is larger than the dimension of the mean.");
	}
	return Vec4f(mean.at<float>(vertexIndex), mean.at<float>(vertexIndex+1), mean.at<float>(vertexIndex+2), 1.0f);
}

Mat PcaModel::drawSample(float sigma/*=1.0f*/)
{
	std::normal_distribution<float> distribution(0.0f, sigma); // this constructor takes the stddev

	vector<float> alphas(getNumberOfPrincipalComponents());

	for (auto&& a : alphas) {
		a = distribution(engine);
	}

	return drawSample(alphas);
}

Mat PcaModel::drawSample(vector<float> coefficients)
{
	// Fill the rest with zeros if not all coefficients are given:
	if (coefficients.size() < getNumberOfPrincipalComponents()) {
		coefficients.resize(getNumberOfPrincipalComponents());
	}
	Mat alphas(coefficients);

	Mat modelSample = mean + normalisedPcaBasis * alphas;

	return modelSample;
}

cv::Mat PcaModel::getNormalisedPcaBasis() const
{
	return normalisedPcaBasis.clone();
}

cv::Mat PcaModel::getNormalisedPcaBasis(int vertexId) const
{
	vertexId *= 3; // the basis is stored in the format [x y z x y z ...]
	return normalisedPcaBasis.rowRange(vertexId, vertexId + 3);
}

cv::Mat PcaModel::getUnnormalisedPcaBasis() const
{
	return unnormalisedPcaBasis.clone();
}

cv::Mat PcaModel::getUnnormalisedPcaBasis(int vertexId) const
{
	vertexId *= 3; // the basis is stored in the format [x y z x y z ...]
	return unnormalisedPcaBasis.rowRange(vertexId, vertexId + 3);
}

float PcaModel::getEigenvalue(int index) const
{
	return eigenvalues.at<float>(index);
}

cv::Mat normalisePcaBasis(cv::Mat unnormalisedBasis, cv::Mat eigenvalues)
{
	Mat normalisedPcaBasis(unnormalisedBasis.size(), unnormalisedBasis.type()); // empty matrix with the same dimensions
	Mat sqrtOfEigenvalues = eigenvalues.clone();
	for (int i = 0; i < eigenvalues.rows; ++i)	{
		sqrtOfEigenvalues.at<float>(i) = std::sqrt(eigenvalues.at<float>(i));
	}
	// Normalise the basis: We multiply each eigenvector (i.e. each column) with the square root of its corresponding eigenvalue
	for (int basis = 0; basis < unnormalisedBasis.cols; ++basis) {
		Mat normalisedEigenvector = unnormalisedBasis.col(basis).mul(sqrtOfEigenvalues.at<float>(basis));
		normalisedEigenvector.copyTo(normalisedPcaBasis.col(basis));
	}
	
	return normalisedPcaBasis;
}

cv::Mat unnormalisePcaBasis(cv::Mat normalisedBasis, cv::Mat eigenvalues)
{
	Mat unnormalisedBasis(normalisedBasis.size(), normalisedBasis.type()); // empty matrix with the same dimensions
	Mat oneOverSqrtOfEigenvalues = eigenvalues.clone();
	for (int i = 0; i < eigenvalues.rows; ++i)	{
		oneOverSqrtOfEigenvalues.at<float>(i) = 1.0f / std::sqrt(eigenvalues.at<float>(i));
	}
	// De-normalise the basis: We multiply each eigenvector (i.e. each column) with 1 over the square root of its corresponding eigenvalue
	for (int basis = 0; basis < normalisedBasis.cols; ++basis) {
		Mat unnormalisedEigenvector = normalisedBasis.col(basis).mul(oneOverSqrtOfEigenvalues.at<float>(basis));
		unnormalisedEigenvector.copyTo(unnormalisedBasis.col(basis));
	}

	return unnormalisedBasis;
}

	} /* namespace morphablemodel */
} /* namespace eos */
