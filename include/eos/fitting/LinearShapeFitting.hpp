/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/LinearShapeFitting.hpp
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

#ifndef LINEARSHAPEFITTING_HPP_
#define LINEARSHAPEFITTING_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include "boost/optional.hpp"

#include <vector>

namespace eos {
	namespace fitting {

/**
 * Fits the shape of a Morphable Model to given 2D landmarks (i.e. estimates the maximum likelihood solution of the shape coefficients) as proposed in [1].
 * It's a linear, closed-form solution fitting of the shape, with regularisation (prior towards the mean).
 *
 * [1] O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.
 *
 * Note: Using less than the maximum number of coefficients to fit is not thoroughly tested yet and may contain an error.
 * Note: Returns coefficients following standard normal distribution (i.e. all have similar magnitude). Why? Because we fit using the normalised basis?
 * Note: The standard deviations given should be a vector, i.e. different for each landmark. This is not implemented yet.
 *
 * @param[in] morphableModel The Morphable Model whose shape (coefficients) are estimated.
 * @param[in] affineCameraMatrix A 3x4 affine camera matrix from world to clip-space (should probably be of type CV_32FC1 as all our calculations are done with float).
 * @param[in] landmarks 2D landmarks from an image, given in clip-coordinates.
 * @param[in] vertexIds The vertex ids in the model that correspond to the 2D points.
 * @param[in] lambda The regularisation parameter (weight of the prior towards the mean).
 * @param[in] numCoefficientsToFit How many shape-coefficients to fit (all others will stay 0). Not tested thoroughly.
 * @param[in] detectorStandardDeviation The standard deviation of the 2D landmarks given (e.g. of the detector used).
 * @param[in] modelStandardDeviation The standard deviation of the 3D vertex points in the 3D model.
 * @return The estimated shape-coefficients (alphas).
 */
std::vector<float> fitShapeToLandmarksLinear(morphablemodel::MorphableModel morphableModel, cv::Mat affineCameraMatrix, std::vector<cv::Vec2f> landmarks, std::vector<int> vertexIds, float lambda=20.0f, boost::optional<int> numCoefficientsToFit=boost::optional<int>(), boost::optional<float> detectorStandardDeviation=boost::optional<float>(), boost::optional<float> modelStandardDeviation=boost::optional<float>());

	} /* namespace fitting */
} /* namespace eos */

#endif /* LINEARSHAPEFITTING_HPP_ */
