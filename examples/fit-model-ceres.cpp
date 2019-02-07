/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model-ceres.cpp
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
#define GLM_FORCE_UNRESTRICTED_GENTYPE

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/ceres_nonlinear.hpp"
#include "eos/morphablemodel/Blendshape.hpp"

using namespace eos;
using namespace ceres;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using Eigen::Vector2f;
using eos::core::IndexedLandmark;
using eos::core::IndexedLandmarkCollection;
using eos::core::Landmark;
using eos::core::LandmarkCollection;

#ifndef EOS_CERES_EXAMPLE_USE_PERSPECTIVE
#define EOS_CERES_EXAMPLE_USE_PERSPECTIVE true
#endif

#ifndef EOS_CERES_EXAMPLE_SHAPES_NUM
#define EOS_CERES_EXAMPLE_SHAPES_NUM 63
#endif

#ifndef EOS_CERES_EXAMPLE_BLENDSHAPES_NUM
#define EOS_CERES_EXAMPLE_BLENDSHAPES_NUM 6
#endif

#ifndef EOS_CERES_EXAMPLE_COLOR_COEFFS_NUM
#define EOS_CERES_EXAMPLE_COLOR_COEFFS_NUM 10
#endif

namespace eos {
namespace ceres_example {
struct HelpCallException : public std::exception
{
    const char* what() const noexcept override
    {
        return "User called help";
    }
};

struct CliArguments
{
    fs::path blendshapesfile, contourfile, imagefile, landmarksfile, mappingsfile, modelfile, outputfile;
};

CliArguments parse_cli_arguments(int argc, char* argv[])
{
    fs::path blendshapesfile, contourfile, imagefile, landmarksfile, mappingsfile, modelfile, outputfile;

    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
                desc.add_options()
                        ("help,h", "display the help message")
                        ("model,m", po::value<fs::path>(&modelfile)->required()->default_value(
                                "../share/sfm_shape_3448.bin"),
                         "a Morphable Model, containing a shape and albedo model, stored as cereal BinaryArchive")
                        ("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value(
                                "../share/expression_blendshapes_3448.bin"),
                         "file with blendshapes")
                        ("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0010.png"),
                         "an input image")
                        ("landmarks,l",
                         po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
                         "2D landmarks for the image, in ibug .pts format")
                        ("mapping,p",
                         po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
                         "landmark identifier to model vertex number mapping")
                        ("model-contour,c",
                         po::value<fs::path>(&contourfile)->required()->default_value(
                                 "../share/sfm_model_contours.json"),
                         "file with model contour indices")
                        ("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
                         "basename for the output obj file");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            std::cout << "Usage: fit-model-ceres [options]" << std::endl;
            std::cout << desc;
            throw HelpCallException();
        }
        po::notify(vm);
    } catch (const po::error& e)
    {
        std::string error_text =
            static_cast<std::string>("Error while parsing command-line arguments: ") + e.what() + '\n';
        std::cout << error_text;
        std::cout << "Use --help to display a list of options." << std::endl;
        throw std::invalid_argument(error_text);
    }

    return CliArguments{blendshapesfile, contourfile, imagefile, landmarksfile,
                        mappingsfile,    modelfile,   outputfile};
}

struct FittingData
{
    fitting::ModelContour model_contour;
    fitting::ContourLandmarks ibug_contour;
    Mat image;
    LandmarkCollection<Vector2f> landmarks;
    morphablemodel::MorphableModel morphable_model;
    core::LandmarkMapper landmark_mapper;
    std::vector<morphablemodel::Blendshape> blendshapes;
};

FittingData read_fitting_data(const std::string& blendshapesfile, const std::string& contourfile,
                              const std::string& imagefile, const std::string& landmarksfile,
                              const std::string& mappingsfile, const std::string& modelfile,
                              const std::string& outputfile)
{
    fitting::ModelContour model_contour;
    fitting::ContourLandmarks ibug_contour;
    if (contourfile.empty())
    {
        model_contour = fitting::ModelContour();
        ibug_contour = fitting::ContourLandmarks();
    } else
    {
        model_contour = fitting::ModelContour::load(contourfile);
        try
        {
            ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
        } catch (const std::runtime_error& e)
        {
            const std::string error_text =
                static_cast<std::string>("Error reading the contour mappings file: ") + e.what() + '\n';
            std::cout << error_text << std::flush;
            throw std::runtime_error(error_text);
        }
    }

    Mat image = cv::imread(imagefile);
    LandmarkCollection<Vector2f> landmarks;
    try
    {
        landmarks = core::read_pts_landmarks(landmarksfile);
    } catch (const std::runtime_error& e)
    {
        const std::string error_text =
            static_cast<std::string>("Error reading the landmarks: ") + e.what() + '\n';
        std::cout << error_text << std::flush;
        throw std::runtime_error(error_text);
    }
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error& e)
    {
        const std::string error_text =
            static_cast<std::string>("Error loading the Morphable Model: ") + e.what() + '\n';
        std::cout << error_text << std::flush;
        throw std::runtime_error(error_text);
    }

    // Note: Actually it's a required argument, so it'll never be empty.
    core::LandmarkMapper landmark_mapper =
        mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

    std::vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

    return FittingData{model_contour,   ibug_contour,    image,      landmarks,
                       morphable_model, landmark_mapper, blendshapes};
}

template <typename LandmarkType>
void draw_landmarks(Mat& image, const IndexedLandmarkCollection<LandmarkType>& landmarks,
                    const cv::Scalar& color = {0.0, 255.0, 255.0})
{
    for (const auto& landmark : landmarks)
    {
        cv::rectangle(image, cv::Point2d(landmark.coordinates[0] - 2.0f, landmark.coordinates[1] - 2.0f),
                      cv::Point2d(landmark.coordinates[0] + 2.0f, landmark.coordinates[1] + 2.0f), color);
    }
}

template <typename LandmarkType>
void draw_mesh_vertices(Mat& image, const core::Mesh& mesh,
                        const IndexedLandmarkCollection<LandmarkType>& landmarks,
                        const glm::dmat4x4& tr_matrix, const glm::dmat4x4& projection_matrix,
                        const glm::dvec4& viewport, const cv::Scalar& color = {0.0f, 0.0f, 255.0f})
{
    for (const auto& landmark : landmarks)
    {
        const auto& vertex = mesh.vertices[landmark.model_index];
        glm::dvec3 point_3d(vertex[0], vertex[1], vertex[2]); // The 3D model point
        glm::dvec3 projected_point = glm::project(point_3d, tr_matrix, projection_matrix, viewport);
        cv::circle(image, cv::Point2d(projected_point.x, projected_point.y), 3, color); // red
    }
}

core::Image3u cv_mat_to_image3u(const cv::Mat& cv_image) {
    core::Image3u image(cv_image.rows, cv_image.cols);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const auto &pixel = cv_image.at<cv::Vec3b>(y, x);
            image(y, x) = core::Image3u::pixel_type(pixel.val[0], pixel.val[1], pixel.val[2]);
        }
    }

    return image;
}

} // namespace ceres_example
} // namespace eos

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const std::array<T, N>& a)
{
    if (!a.empty())
    {
        out << '[';
        std::copy(a.begin(), a.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
};

/**
 * Single and multi-image non-linear model fitting with Ceres example.
 *
 * NOTE: The ImageCost cost function requires the "full" 3DMM with the
 * albedo model. It can be acquired from CVSSP - see the GitHub main page.
 * If you don't currently have it, and still want to try the Ceres fitting,
 * the ImageCost can just be removed.
 */
int main(int argc, char* argv[])
{
    // Read cli arguments
    ceres_example::CliArguments cli_arguments;
    try
    {
        cli_arguments = ceres_example::parse_cli_arguments(argc, argv);
    } catch (const ceres_example::HelpCallException& e)
    {
        return EXIT_SUCCESS;
    } catch (const std::invalid_argument& e)
    {
        return EXIT_FAILURE;
    }

    // Read all data from disk
    ceres_example::FittingData fitting_data;
    try
    {
        fitting_data = ceres_example::read_fitting_data(
            cli_arguments.blendshapesfile.string(), cli_arguments.contourfile.string(),
            cli_arguments.imagefile.string(), cli_arguments.landmarksfile.string(),
            cli_arguments.mappingsfile.string(), cli_arguments.modelfile.string(),
            cli_arguments.outputfile.string());
    } catch (const std::runtime_error& e)
    {
        return EXIT_FAILURE;
    }

    const auto& model_contour = fitting_data.model_contour;
    const auto& ibug_contour = fitting_data.ibug_contour;
    const auto& cv_image = fitting_data.image;
    const auto& morphable_model = fitting_data.morphable_model;
    const auto& landmark_mapper = fitting_data.landmark_mapper;
    const auto& blendshapes = fitting_data.blendshapes;

    auto image = eos::ceres_example::cv_mat_to_image3u(cv_image);

    // These will be the 2D image points and their corresponding 3D vertex id's used for the fitting
    auto& landmarks = fitting_data.landmarks;

    auto landmarks_definitions = morphable_model.get_landmark_definitions();
    auto* landmarks_definitions_ptr =
        landmarks_definitions.has_value() ? &landmarks_definitions.value() : nullptr;
    auto indexed_landmarks = landmark_mapper.get_indexed_landmarks(landmarks, landmarks_definitions_ptr);

    google::InitGoogleLogging(argv[0]); // Ceres logging initialisation
    std::stringstream fitting_log;

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    auto start = std::chrono::steady_clock::now();

    auto model_fitter =
        fitting::ModelFitter<EOS_CERES_EXAMPLE_SHAPES_NUM, EOS_CERES_EXAMPLE_BLENDSHAPES_NUM,
                             EOS_CERES_EXAMPLE_COLOR_COEFFS_NUM>(&morphable_model, &blendshapes);
    auto camera = fitting::PerspectiveCameraParameters(image.width(), image.height());
    model_fitter.add_camera_cost_function(camera, indexed_landmarks);
    model_fitter.block_shapes_fitting();
    model_fitter.block_blendshapes_fitting();
    model_fitter.block_fov_fitting(camera, 60.0);

    auto solver_summary = model_fitter.solve();

    auto end = std::chrono::steady_clock::now();
    // Log fitting report
    std::cout << solver_summary.BriefReport() << std::endl;

    // Draw the mean-face landmarks projected using the estimated camera:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:

    Mat outimg = cv_image.clone();
    ceres_example::draw_mesh_vertices(outimg, morphable_model.get_mean(), indexed_landmarks,
                                      camera.calculate_translation_matrix() *
                                          camera.calculate_rotation_matrix(),
                                      camera.calculate_projection_matrix(), camera.get_viewport());
    ceres_example::draw_landmarks(outimg, indexed_landmarks);

    auto camera_euler_rotation = camera.get_euler_rotation();
    fitting_log << "Pose fit with mean shape:\tYaw " << glm::degrees(camera_euler_rotation[1]) << ", Pitch "
                << glm::degrees(camera_euler_rotation[0]) << ", Roll "
                << glm::degrees(camera_euler_rotation[2]) << "; t & f: " << camera.translation_and_intrinsics
                << '\n'
                << "Ceres took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms."
                << std::endl;

    const auto& outputfile = cli_arguments.outputfile;
    auto new_path = outputfile.parent_path() / fs::path(outputfile.stem().string() + "_pos_only");
    new_path.replace_extension(".png");
    cv::imwrite(new_path.string(), outimg);

    // Contour fitting:
    if (!fitting_data.ibug_contour.empty())
    {
        auto indexed_contours = model_fitter.estimate_contours(camera, fitting_data.ibug_contour,
                                                               fitting_data.model_contour, landmarks);
        indexed_landmarks.insert(indexed_landmarks.end(), indexed_contours.begin(), indexed_contours.end());
        ceres_example::draw_landmarks(outimg, indexed_landmarks); // yellow: subset of the detected LMs that
                                                                  // we use (draw with contour landmarks)
    }

    // Full fitting - Estimate shape and pose, given the previous pose estimate:
    start = std::chrono::steady_clock::now();

    model_fitter.reset_problem();
    model_fitter.add_camera_cost_function(camera, indexed_landmarks);
    model_fitter.block_fov_fitting(camera, 60.0);

    model_fitter.add_shape_prior_cost_function();
    model_fitter.add_blendshape_prior_cost_function();

    // Colour model fitting (this needs a Morphable Model with colour (albedo) model, see note above main()):
    Eigen::VectorXf color_instance;
    darray<EOS_CERES_EXAMPLE_COLOR_COEFFS_NUM> colour_coefficients;
    if (morphable_model.has_color_model())
    {
        // Add a residual for each vertex:
        model_fitter.add_image_cost_function(camera, image);
        model_fitter.add_image_prior_cost_function();

        color_instance = morphable_model.get_color_model().draw_sample(colour_coefficients);
    } else
    {
        std::cout << "The MorphableModel used does not contain a colour (albedo) model. No ImageCost will be "
                     "applied."
                  << std::endl;
        color_instance = Eigen::VectorXf();
    }

    solver_summary = model_fitter.solve();
    std::cout << solver_summary.BriefReport() << std::endl;
    end = std::chrono::steady_clock::now();

    // Draw the landmarks projected using all estimated parameters:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:

    auto points = model_fitter.calculate_estimated_points_positions();

    core::Mesh mesh = morphablemodel::sample_to_mesh(
        points, color_instance, morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates(),
        morphable_model.get_texture_triangle_indices());

    ceres_example::draw_mesh_vertices(
        outimg, morphable_model.get_mean(), indexed_landmarks,
        camera.calculate_translation_matrix() * camera.calculate_rotation_matrix(),
        camera.calculate_projection_matrix(), camera.get_viewport(), {0.0f, 76.0f, 255.0f}); // orange

    camera_euler_rotation = camera.get_euler_rotation();
    fitting_log << "Final fit:\t\t\tYaw " << glm::degrees(camera_euler_rotation[1]) << ", Pitch "
                << glm::degrees(camera_euler_rotation[0]) << ", Roll "
                << glm::degrees(camera_euler_rotation[2]) << "; t & f: " << camera.translation_and_intrinsics
                << std::endl;
    fitting_log << "Ceres took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms."
                << std::endl;

    std::cout << fitting_log.str();

    new_path = outputfile;
    new_path.replace_extension(".obj");
    core::write_obj(mesh, new_path.string());

    new_path.replace_extension(".png");
    cv::imwrite(new_path.string(), outimg);

    return EXIT_SUCCESS;
}
