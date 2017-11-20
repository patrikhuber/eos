/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/+eos/+fitting/private/fitting.cpp
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
#include "eos/fitting/fitting.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"

#include "mexplus_eigen.hpp"
#include "mexplus_eos_types.hpp"

#include "mexplus.h"

#include "Eigen/Core"

#include "mex.h"
//#include "matrix.h"

#include <optional>
#include <string>

using namespace eos;
using namespace mexplus;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    using std::string;
    // Check for proper number of input and output arguments:
    if (nrhs != 12)
    {
        mexErrMsgIdAndTxt("eos:fitting:nargin", "fit_shape_and_pose requires 12 input arguments.");
    }
    if (nlhs != 2)
    {
        mexErrMsgIdAndTxt("eos:fitting:nargout", "fit_shape_and_pose returns two output arguments.");
    }

    InputArguments input(nrhs, prhs, 12);
    const auto morphablemodel_file = input.get<string>(0);
    const auto blendshapes_file = input.get<string>(1);
    const auto landmarks_in = input.get<Eigen::MatrixXd>(2);
    const auto mapper_file = input.get<string>(3);
    const auto image_width = input.get<int>(4);
    const auto image_height = input.get<int>(5);
    const auto edgetopo_file = input.get<string>(6);
    const auto contour_lms_file = input.get<string>(7);
    const auto model_cnt_file = input.get<string>(8);
    const auto num_iterations = input.get<int>(9);
    const auto num_shape_coeffs = input.get<int>(10);
    const auto lambda = input.get<double>(11);

    if (landmarks_in.rows() != 68)
    {
        mexErrMsgIdAndTxt(
            "eos:fitting:argin",
            "Given landmarks must be a 68 x 2 vector with ibug landmarks, in the order from 1 to 68.");
    }
    // Convert the landmarks (given as matrix in Matlab) to a LandmarkCollection:
    core::LandmarkCollection<Eigen::Vector2f> landmarks;
    for (int i = 0; i < 68; ++i)
    {
        landmarks.push_back(core::Landmark<Eigen::Vector2f>{
            std::to_string(i + 1), Eigen::Vector2f(landmarks_in(i, 0), landmarks_in(i, 1))});
    }

    // Load everything:
    const auto morphable_model = morphablemodel::load_model(morphablemodel_file);
    const auto blendshapes = morphablemodel::load_blendshapes(blendshapes_file);
    const core::LandmarkMapper landmark_mapper(mapper_file);
    const auto edge_topology = morphablemodel::load_edge_topology(edgetopo_file);
    const auto contour_landmarks = fitting::ContourLandmarks::load(contour_lms_file);
    const auto model_contour = fitting::ModelContour::load(model_cnt_file);
    const std::optional<int> num_shape_coefficients_to_fit =
        num_shape_coeffs == -1 ? std::nullopt : std::optional<int>(num_shape_coeffs);

    // Now do the actual fitting:
    core::Mesh mesh;
    fitting::RenderingParameters rendering_parameters;
    std::tie(mesh, rendering_parameters) = fitting::fit_shape_and_pose(
        morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height, edge_topology,
        contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda);

    // Return the mesh and the rendering_parameters:
    OutputArguments output(nlhs, plhs, 2);
    output.set(0, mesh);
    output.set(1, rendering_parameters);
};
