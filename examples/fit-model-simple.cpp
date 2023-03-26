/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model-simple.cpp
 *
 * Copyright 2015 Patrik Huber
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
#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/texture_extraction.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using Eigen::Vector2f;
using Eigen::Vector4f;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 *
 * First, the 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper. Then, an orthographic camera
 * is estimated, and then, using this camera matrix, the shape is fitted
 * to the landmarks.
 */
int main(int argc, char* argv[])
{
    string modelfile, imagefile, landmarksfile, mappingsfile, outputbasename;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<string>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
                "a Morphable Model stored as cereal BinaryArchive")
            ("image,i", po::value<string>(&imagefile)->required()->default_value("data/image_0010.png"),
                "an input image")
            ("landmarks,l", po::value<string>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
                "2D landmarks for the image, in ibug .pts format")
            ("mapping,p", po::value<string>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
                "landmark identifier to model vertex number mapping")
            ("output,o", po::value<string>(&outputbasename)->required()->default_value("out"),
                "basename for the output rendering and obj files");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: fit-model-simple [options]" << endl;
            cout << desc;
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    } catch (const po::error& e)
    {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        return EXIT_FAILURE;
    }

    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
    Mat image = cv::imread(imagefile);
    LandmarkCollection<Eigen::Vector2f> landmarks;
    try
    {
        landmarks = core::read_pts_landmarks(landmarksfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error reading the landmarks: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    // The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    core::LandmarkMapper landmark_mapper;
    try
    {
        landmark_mapper = core::LandmarkMapper(mappingsfile);
    } catch (const std::exception& e)
    {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Draw the loaded landmarks:
    Mat outimg = image.clone();
    for (auto&& lm : landmarks)
    {
        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
                      cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
    }

    // These will be the final 2D and 3D points used for the fitting:
    vector<Vector4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices;    // their vertex indices
    vector<Vector2f> image_points; // the corresponding 2D landmark points

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for (int i = 0; i < landmarks.size(); ++i)
    {
        const auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        const int vertex_idx = std::stoi(converted_name.value());
        const auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
        model_points.emplace_back(vertex.homogeneous());
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    fitting::ScaledOrthoProjectionParameters pose =
        fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image.rows);
    fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);

    // The 3D head pose can be recovered as follows - the function returns an Eigen::Vector3f with yaw, pitch,
    // and roll angles:
    const float yaw_angle = rendering_params.get_yaw_pitch_roll()[0];

    // Estimate the shape coefficients by fitting the shape to the landmarks:
    const Eigen::Matrix<float, 3, 4> affine_from_ortho =
        fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
    const vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(
        morphable_model.get_shape_model(), affine_from_ortho, image_points, vertex_indices);

    // Obtain the full mesh with the estimated coefficients:
    const core::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

    // Extract the texture from the image using given mesh and camera parameters:
    const core::Image4u texturemap =
        render::extract_texture(mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                                render::ProjectionType::Orthographic, core::from_mat_with_alpha(image));

    // Save the mesh as textured obj:
    fs::path outputfile = outputbasename + ".obj";
    core::write_textured_obj(mesh, outputfile.string());

    // And save the texture map:
    outputfile.replace_extension(".texture.png");
    cv::imwrite(outputfile.string(), core::to_mat(texturemap));

    cout << "Finished fitting and wrote result mesh and texture to files with basename "
         << outputfile.stem().stem() << "." << endl;

    return EXIT_SUCCESS;
}
