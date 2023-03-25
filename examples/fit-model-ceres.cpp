/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model-ceres.cpp
 *
 * Copyright 2016-2023 Patrik Huber
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
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/core/math.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/ceres_nonlinear.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/rotation_angles.hpp"
#include "eos/render/matrix_projection.hpp"

#include "Eigen/Core"

#include "ceres/ceres.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include <chrono>
#include <iostream>
#include <vector>

using namespace eos;
using namespace ceres;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using Eigen::Vector2f;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// print a vector:
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
    if (!v.empty())
    {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
};

/**
 * Single image non-linear model fitting with Ceres example.
 *
 * NOTE: The VertexColorCost cost function requires a 3DMM with per-vertex albedo (or colour) model. You
 * could, for example, acquire:
 * - the full Surrey Face Model (see "The Surrey Face Model" in the eos README.md)
 * - the 4D Face Model (4DFM) from www.4dface.io
 * - the Basel Face Model (BFM).
 * If you don't currently have a full 3DMM, and still want to try the Ceres fitting, the VertexColorCost can
 * just be removed from the main() function below.
 */
int main(int argc, char* argv[])
{
    fs::path modelfile, imagefile, landmarksfile, mappingsfile, contourfile, blendshapesfile, outputfile;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_3448.bin"),
                "a Morphable Model, containing a shape and albedo model, stored as cereal BinaryArchive")
            ("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
                "file with blendshapes")
            ("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0010.png"),
                "an input image")
            ("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
                "2D landmarks for the image, in ibug .pts format")
            ("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
                "landmark identifier to model vertex number mapping")
            ("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/sfm_model_contours.json"),
                "file with model contour indices")
            ("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
                "basename for the output obj file");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: fit-model-ceres [options]" << endl;
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

    google::InitGoogleLogging(argv[0]); // Ceres logging initialisation

    fitting::ModelContour model_contour =
        contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());

    fitting::ContourLandmarks ibug_contour;
    try
    {
        ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());
    } catch (const std::runtime_error& e)
    {
        cout << "Error reading the contour mappings file: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
    Mat image = cv::imread(imagefile.string());
    LandmarkCollection<Eigen::Vector2f> landmarks;
    try
    {
        landmarks = core::read_pts_landmarks(landmarksfile.string());
    } catch (const std::runtime_error& e)
    {
        cout << "Error reading the landmarks: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile.string());
    } catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Note: Actually it's a required argument, so it'll never be empty.
    core::LandmarkMapper landmark_mapper =
        mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile.string());

    std::vector<eos::morphablemodel::Blendshape> blendshapes =
        eos::morphablemodel::load_blendshapes(blendshapesfile.string());

    // Draw the loaded landmarks (blue):
    Mat outimg = image.clone();
    for (const auto& lm : landmarks)
    {
        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
                      cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
    }

    // These will be the 2D image points and their corresponding 3D vertex id's used for the fitting:
    vector<Vector2f> image_points; // the 2D landmark points
    vector<int> vertex_indices;    // their corresponding vertex indices

    // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
    for (int i = 0; i < landmarks.size(); ++i)
    {
        const auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        const int vertex_idx = std::stoi(converted_name.value());
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    std::stringstream fitting_log;
    auto start = std::chrono::steady_clock::now();

    // The following have to be constexpr because they're used in template parameters in this implementation:
    constexpr int num_shape_coeffs = 50;
    constexpr int num_blendshape_coeffs = 6;
    constexpr int num_color_coeffs = 50;

    // Weights for the different cost function terms:
    // Note: To find good weights, ceres::ResidualBlockId can be used to store the IDs to each added residual
    // blocks. Then, one can use ceres::Problem::EvaluateOptions and ceres::Problem::Evaluate(...) to retrieve
    // the values of each individual term. We don't do that here for simplicity reasons. See the Ceres
    // documentation for details, and https://github.com/patrikhuber/eos/issues/348.
    const double landmark_cost_weight = 100.0;
    const double shape_prior_cost_weight = 500000.0;
    const double blendshapes_prior_cost_weight = 25.0;
    const double color_prior_cost_weight = 500000.0;
    const double vertex_color_cost_weight = 1.0;

    // The model parameters that we're estimating:
    using ModelCoefficients = vector<double>;
    ModelCoefficients shape_coefficients(num_shape_coeffs, 0.0);
    ModelCoefficients blendshape_coefficients(num_blendshape_coeffs, 0.0);
    ModelCoefficients color_coefficients(num_color_coeffs, 0.0);

    // Parameters for the perspective projection: A rotation quaternion, [t_x, t_y, t_z], and fov_y (field of
    // view).
    // The origin is assumed at center of image, and no lens distortions.
    // Note: Actually, we estimate the model-view matrix and not the camera position. But one defines the
    // other.
    Eigen::Quaterniond camera_rotation(1.0, 0.0, 0.0, 0.0); // The c'tor takes wxyz. Storage is xyzw.
    Eigen::Vector3d camera_translation(0.0, 0.0, -400.0);   // Move the model back (along the -z axis)
    double fov_y = core::radians(45.0);

    // Set up just a landmark cost and optimise for rotation and translation, with fov_y and the 3DMM
    // parameters fixed (i.e. using the mean face):
    Problem camera_costfunction;
    for (int i = 0; i < image_points.size(); ++i)
    {
        auto landmark_cost =
            fitting::PerspectiveProjectionLandmarkCost::Create<num_shape_coeffs, num_blendshape_coeffs>(
                morphable_model.get_shape_model(), blendshapes, image_points[i], vertex_indices[i],
                image.cols, image.rows);
        ScaledLoss* landmark_cost_scaled_loss =
            new ScaledLoss(nullptr, landmark_cost_weight, ceres::TAKE_OWNERSHIP);
        camera_costfunction.AddResidualBlock(landmark_cost, landmark_cost_scaled_loss,
                                             &camera_rotation.coeffs()[0], &camera_translation.data()[0],
                                             &fov_y, &shape_coefficients[0], &blendshape_coefficients[0]);
    }
    // Keep the model coeffs constant (i.e. all at zero):
    camera_costfunction.SetParameterBlockConstant(&shape_coefficients[0]);
    camera_costfunction.SetParameterBlockConstant(&blendshape_coefficients[0]);
    // Keep the fov_y constant too:
    camera_costfunction.SetParameterBlockConstant(&fov_y);
    // Keep t_z negative (i.e. we want to look at the model from the front):
    camera_costfunction.SetParameterUpperBound(&camera_translation[0], 2,
                                               -std::numeric_limits<double>::epsilon());
    // Keep the fov_y (in radians) > 0 (but we're not optimising for it now):
    // camera_costfunction.SetParameterLowerBound(&fov_y, 0, 0.01);

    EigenQuaternionManifold* quaternion_manifold = new EigenQuaternionManifold;
    camera_costfunction.SetManifold(&camera_rotation.coeffs()[0], quaternion_manifold);

    // Set up the solver, and optimise:
    Solver::Options solver_options;
    // solver_options.linear_solver_type = ITERATIVE_SCHUR;
    // solver_options.num_threads = 8;
    solver_options.minimizer_progress_to_stdout = true;
    // solver_options.max_num_iterations = 100;
    Solver::Summary solver_summary;
    Solve(solver_options, &camera_costfunction, &solver_summary);
    std::cout << solver_summary.BriefReport() << "\n";
    auto end = std::chrono::steady_clock::now();

    // Draw the mean-face landmarks projected using the estimated camera:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:
    Eigen::Matrix4d model_view_mtx = Eigen::Matrix4d::Identity();
    model_view_mtx.block<3, 3>(0, 0) = camera_rotation.normalized().toRotationMatrix();
    model_view_mtx.col(3).head<3>() = camera_translation;
    const double aspect = static_cast<double>(image.cols) / image.rows;
    auto projection_mtx = render::perspective(fov_y, aspect, 0.1, 1000.0);
    // Todo: use get_opencv_viewport() from nonlin_cam_esti.hpp.
    const Eigen::Vector4d viewport(0, image.rows, image.cols, -image.rows); // OpenCV convention

    const auto& mean_mesh = morphable_model.get_mean();
    for (auto idx : vertex_indices)
    {
        const auto& point_3d = mean_mesh.vertices[idx];
        const auto projected_point =
            render::project<double>(point_3d.cast<double>(), model_view_mtx, projection_mtx, viewport);
        cv::circle(outimg, cv::Point2f(projected_point.x(), projected_point.y()), 3,
                   {0.0f, 0.0f, 255.0f}); // red
    }
    // Draw the subset of the detected landmarks that we use in the fitting (yellow):
    for (const auto& lm : image_points)
    {
        cv::circle(outimg, cv::Point2f(lm(0), lm(1)), 3, {0.0f, 255.0f, 255.0f});
    }

    fitting_log << "Pose-only fitting took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms.\n";

    // Now add contour fitting:
    // Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
    // Note: It would be a good idea to re-compute correspondences during the fitting, as opposed to only
    // computing them once at the start using the mean face.
    const float yaw_angle = core::degrees(
        fitting::tait_bryan_angles(camera_rotation.normalized().toRotationMatrix(), 1, 0, 2)[0]);
    vector<Vector2f> image_points_contour; // the 2D landmark points
    vector<int> vertex_indices_contour;    // their corresponding 3D vertex indices
    // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
    std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
        fitting::get_contour_correspondences(landmarks, ibug_contour, model_contour, yaw_angle,
                                             morphable_model.get_mean(), model_view_mtx.cast<float>(),
                                             projection_mtx.cast<float>(), viewport.cast<float>());
    using eos::fitting::concat;
    vertex_indices = concat(vertex_indices, vertex_indices_contour);
    image_points = concat(image_points, image_points_contour);
    // Note: We could also fit the occluding ("away-facing") contour, like in fit_shape_and_pose() (see
    // fitting.hpp), but we don't do that here for simplicity reasons.

    // Full fitting - Estimate shape and pose, given the previous pose estimate:
    start = std::chrono::steady_clock::now();
    Problem fitting_costfunction;
    // Landmark cost:
    for (int i = 0; i < image_points.size(); ++i)
    {
        auto landmark_cost =
            fitting::PerspectiveProjectionLandmarkCost::Create<num_shape_coeffs, num_blendshape_coeffs>(
                morphable_model.get_shape_model(), blendshapes, image_points[i], vertex_indices[i],
                image.cols, image.rows);
        ScaledLoss* landmark_cost_scaled_loss =
            new ScaledLoss(nullptr, landmark_cost_weight, ceres::TAKE_OWNERSHIP);
        fitting_costfunction.AddResidualBlock(landmark_cost, landmark_cost_scaled_loss,
                                              &camera_rotation.coeffs()[0], &camera_translation.data()[0],
                                              &fov_y, &shape_coefficients[0], &blendshape_coefficients[0]);
    }
    // Prior and bounds for the shape coefficients:
    CostFunction* shape_prior_cost = fitting::NormCost::Create<num_shape_coeffs>();
    ScaledLoss* shape_prior_scaled_loss =
        new ScaledLoss(nullptr, shape_prior_cost_weight,
                       Ownership::TAKE_OWNERSHIP); // weight was 35.0 previously, but it was inside the
                                                   // residual, now it's outside
    fitting_costfunction.AddResidualBlock(shape_prior_cost, shape_prior_scaled_loss, &shape_coefficients[0]);
    for (int i = 0; i < num_shape_coeffs; ++i)
    {
        fitting_costfunction.SetParameterLowerBound(&shape_coefficients[0], i, -3.0);
        fitting_costfunction.SetParameterUpperBound(&shape_coefficients[0], i, 3.0);
    }
    // Prior and constraints on blendshapes:
    CostFunction* blendshapes_prior_cost = fitting::NormCost::Create<num_blendshape_coeffs>();
    LossFunction* blendshapes_prior_scaled_loss =
        new ScaledLoss(new SoftLOneLoss(1.0), blendshapes_prior_cost_weight,
                       Ownership::TAKE_OWNERSHIP); // weight was 10.0 previously
    fitting_costfunction.AddResidualBlock(blendshapes_prior_cost, blendshapes_prior_scaled_loss,
                                          &blendshape_coefficients[0]);
    for (int i = 0; i < num_blendshape_coeffs; ++i)
    {
        fitting_costfunction.SetParameterLowerBound(&blendshape_coefficients[0], i, 0.0);
        fitting_costfunction.SetParameterUpperBound(&blendshape_coefficients[0], i, 1.0);
    }
    // Keep t_z negative (i.e. we want to look at the model from the front):
    fitting_costfunction.SetParameterUpperBound(&camera_translation[0], 2,
                                                -std::numeric_limits<double>::epsilon());
    // Keep the fov_y (in radians) > 0:
    fitting_costfunction.SetParameterLowerBound(&fov_y, 0, 0.01);
    // Note: We create a new manifold object, since camera_costfunction took ownership of the previous one,
    // and will delete it upon destruction.
    EigenQuaternionManifold* quaternion_manifold_full_fitting = new EigenQuaternionManifold;
    fitting_costfunction.SetManifold(&camera_rotation.coeffs()[0], quaternion_manifold_full_fitting);

    // Colour model cost (this needs a Morphable Model with colour (albedo) model, see note above main()):
    if (!morphable_model.has_color_model())
    {
        cout << "Error: The MorphableModel used does not contain a colour (albedo) model. ImageCost requires "
                "a model that contains a colour PCA model. You may want to use the full Surrey Face Model, a "
                "different morphable model, or remove this section.";
        return EXIT_FAILURE;
    }
    // Add a colour residual for each vertex:
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    const core::Image3u image_eos_rgb = eos::core::from_mat(image);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    for (int i = 0; i < morphable_model.get_shape_model().get_data_dimension() / 3; ++i)
    {
        auto vertex_color_cost =
            fitting::VertexColorCost::Create<num_shape_coeffs, num_blendshape_coeffs, num_color_coeffs>(
                morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(), i,
                image_eos_rgb);
        ScaledLoss* vertex_color_cost_scaled_loss =
            new ScaledLoss(nullptr, vertex_color_cost_weight, ceres::TAKE_OWNERSHIP);
        fitting_costfunction.AddResidualBlock(vertex_color_cost, vertex_color_cost_scaled_loss,
                                              &camera_rotation.coeffs()[0], &camera_translation.data()[0],
                                              &fov_y, &shape_coefficients[0], &blendshape_coefficients[0],
                                              &color_coefficients[0]);
    }
    // Prior and bounds, for the colour coefficients:
    CostFunction* color_prior_cost = fitting::NormCost::Create<num_color_coeffs>();
    ScaledLoss* color_prior_scaled_loss = new ScaledLoss(
        nullptr, color_prior_cost_weight, Ownership::TAKE_OWNERSHIP); // weight was previously 35.0
    fitting_costfunction.AddResidualBlock(color_prior_cost, color_prior_scaled_loss, &color_coefficients[0]);
    for (int i = 0; i < num_color_coeffs; ++i)
    {
        fitting_costfunction.SetParameterLowerBound(&color_coefficients[0], i, -3.0);
        fitting_costfunction.SetParameterUpperBound(&color_coefficients[0], i, 3.0);
    }

    // Set different options for the full fitting:
    // solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    // solver_options.linear_solver_type = ceres::DENSE_QR;
    // solver_options.minimizer_type = ceres::TRUST_REGION; // default I think
    // solver_options.minimizer_type = ceres::LINE_SEARCH;
    solver_options.num_threads = 16; // Make sure to adjust this, if you have fewer (or more) CPU cores.
    // solver_options.minimizer_progress_to_stdout = true;
    // solver_options.max_num_iterations = 100;
    Solve(solver_options, &fitting_costfunction, &solver_summary);
    std::cout << solver_summary.BriefReport() << "\n";
    end = std::chrono::steady_clock::now();

    // Draw the landmarks projected using all estimated parameters:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:
    model_view_mtx = Eigen::Matrix4d::Identity();
    model_view_mtx.block<3, 3>(0, 0) = camera_rotation.normalized().toRotationMatrix();
    model_view_mtx.col(3).head<3>() = camera_translation;
    projection_mtx = render::perspective(fov_y, aspect, 0.1, 1000.0);

    auto vectord_to_vectorf = [](const std::vector<double>& vec) {
        return std::vector<float>(std::begin(vec), std::end(vec));
    };
    const auto shape_coeffs_float = vectord_to_vectorf(shape_coefficients);
    const auto blendshape_coeffs_float = vectord_to_vectorf(blendshape_coefficients);
    const auto color_coeffs_float = vectord_to_vectorf(color_coefficients);

    morphablemodel::MorphableModel morphable_model_with_expressions(
        morphable_model.get_shape_model(), blendshapes, morphable_model.get_color_model(),
        morphable_model.get_landmark_definitions(), morphable_model.get_texture_coordinates(),
        morphable_model.get_texture_triangle_indices());
    const core::Mesh mesh = morphable_model_with_expressions.draw_sample(
        shape_coeffs_float, blendshape_coeffs_float, color_coeffs_float);

    // Draw the vertices that we used in the fitting (orange):
    for (auto idx : vertex_indices)
    {
        const auto& point_3d = mesh.vertices[idx];
        const auto projected_point =
            render::project<double>(point_3d.cast<double>(), model_view_mtx, projection_mtx, viewport);
        cv::circle(outimg, cv::Point2f(projected_point.x(), projected_point.y()), 3,
                   {0.0f, 76.0f, 255.0f}); // orange
    }
    // Draw the corresponding 2D landmarks (now including contour landmarks):
    for (const auto& lm : image_points)
    {
        cv::circle(outimg, cv::Point2f(lm(0), lm(1)), 3, {0.0f, 255.0f, 255.0f}); // yellow
    }
    // Note: Save outimg for debug purposes, or inspect it in e.g. ImageWatch.

    fitting_log << "Full fitting took: "
                << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s.\n";

    cout << fitting_log.str();

    outputfile.replace_extension(".obj");
    core::write_obj(mesh, outputfile.string());

    return EXIT_SUCCESS;
}
