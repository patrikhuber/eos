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
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"
#include "Eigen/Core"
#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/ceres_nonlinear.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"


using namespace eos;
using namespace ceres;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using Eigen::Vector2f;


#ifndef EOS_CERES_USE_PERSPECTIVE
#define EOS_CERES_USE_PERSPECTIVE true
#endif

#ifndef EOS_CERES_SHAPES_NUM
#define EOS_CERES_SHAPES_NUM 63
#endif

#ifndef EOS_CERES_BLENDSHAPES_NUM
#define EOS_CERES_BLENDSHAPES_NUM 6
#endif

#ifndef EOS_CERES_COLOR_COEFFS_NUM
#define EOS_CERES_COLOR_COEFFS_NUM 10
#endif


namespace eos {
    namespace ceres_example {
        struct HelpCallException : public std::exception {
            const char *what() const noexcept override {
                return "User called help";
            }
        };

        struct CliArguments {
            fs::path blendshapesfile,
                     contourfile,
                     imagefile,
                     landmarksfile,
                     mappingsfile,
                     modelfile,
                     outputfile;
        };


        CliArguments parse_cli_arguments(int argc, char *argv[]) {
            fs::path blendshapesfile,
                     contourfile,
                     imagefile,
                     landmarksfile,
                     mappingsfile,
                     modelfile,
                     outputfile;

            try {
                po::options_description desc("Allowed options");
                // clang-format off
                desc.add_options()
                        ("help,h", "display the help message")
                        ("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_3448.bin"),
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
                if (vm.count("help")) {
                    std::cout << "Usage: fit-model-ceres [options]" << std::endl;
                    std::cout << desc;
                    throw HelpCallException();
                }
                po::notify(vm);
            } catch (const po::error &e) {
                std::string error_text = static_cast<std::string>("Error while parsing command-line arguments: ") +
                                         e.what() + '\n';
                std::cout << error_text;
                std::cout << "Use --help to display a list of options." << std::endl;
                throw std::invalid_argument(error_text);
            }

            return CliArguments{blendshapesfile,
                                contourfile,
                                imagefile,
                                landmarksfile,
                                mappingsfile,
                                modelfile,
                                outputfile};
        }

        struct FittingData {
            fitting::ModelContour model_contour;
            fitting::ContourLandmarks ibug_contour;
            Mat image;
            LandmarkCollection<Vector2f> landmarks;
            morphablemodel::MorphableModel morphable_model;
            core::LandmarkMapper landmark_mapper;
            std::vector<eos::morphablemodel::Blendshape> blendshapes;
        };


        FittingData read_fitting_data(const std::string &blendshapesfile,
                                      const std::string &contourfile,
                                      const std::string &imagefile,
                                      const std::string &landmarksfile,
                                      const std::string &mappingsfile,
                                      const std::string &modelfile,
                                      const std::string &outputfile) {
            fitting::ModelContour model_contour;
            fitting::ContourLandmarks ibug_contour;
            if (contourfile.empty()) {
                model_contour = fitting::ModelContour();
                ibug_contour = fitting::ContourLandmarks();
            } else {
                model_contour = fitting::ModelContour::load(contourfile);
                try {
                    ibug_contour = fitting::ContourLandmarks::load(mappingsfile);
                } catch (const std::runtime_error &e) {
                    const std::string error_text =
                            static_cast<std::string>("Error reading the contour mappings file: ") +
                            e.what() + '\n';
                    std::cout << error_text << std::flush;
                    throw std::runtime_error(error_text);
                }
            }

            Mat image = cv::imread(imagefile);
            LandmarkCollection<Vector2f> landmarks;
            try {
                landmarks = core::read_pts_landmarks(landmarksfile);
            } catch (const std::runtime_error &e) {
                const std::string error_text = static_cast<std::string>("Error reading the landmarks: ") +
                                               e.what() + '\n';
                std::cout << error_text << std::flush;
                throw std::runtime_error(error_text);
            }
            morphablemodel::MorphableModel morphable_model;
            try {
                morphable_model = morphablemodel::load_model(modelfile);
            } catch (const std::runtime_error &e) {
                const std::string error_text = static_cast<std::string>("Error loading the Morphable Model: ") +
                                               e.what() + '\n';
                std::cout << error_text << std::flush;
                throw std::runtime_error(error_text);
            }

            // Note: Actually it's a required argument, so it'll never be empty.
            core::LandmarkMapper landmark_mapper =
                    mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

            std::vector<eos::morphablemodel::Blendshape> blendshapes =
                    eos::morphablemodel::load_blendshapes(blendshapesfile);

            return {model_contour,
                    ibug_contour,
                    image,
                    landmarks,
                    morphable_model,
                    landmark_mapper,
                    blendshapes};
        }


        auto get_camera_translation_and_intrinsics() {
            // Parameters for the orthographic projection: [t_x, t_y, frustum_scale]
            // And perspective projection: [t_x, t_y, t_z, fov].
            // Origin is assumed at center of image, and no lens distortions.
            // Note: Actually, we estimate the model-view matrix and not the camera position. But one defines the
            // other.

            darray<num_cam_params(EOS_CERES_USE_PERSPECTIVE)> camera_translation_and_intrinsics;
            if (EOS_CERES_USE_PERSPECTIVE) {
                camera_translation_and_intrinsics[2] = -400.0;              // Move the model back (along the -z axis)
                camera_translation_and_intrinsics[3] = glm::radians(60.0f); // fov
            } else {
                camera_translation_and_intrinsics[2] = 110.0; // frustum_scale
            }
            return camera_translation_and_intrinsics;
        }


        Solver::Options get_solver_options() {
            Solver::Options solver_options;
            solver_options.linear_solver_type = ITERATIVE_SCHUR;
            solver_options.num_threads = 8;
            solver_options.num_linear_solver_threads = 8; // only SPARSE_SCHUR can use this
            solver_options.minimizer_progress_to_stdout = true;
            // solver_options.max_num_iterations = 100;

            return solver_options;
        }

        auto get_translation_matrix(
                const darray<num_cam_params(EOS_CERES_USE_PERSPECTIVE)> &camera_translation_and_intrinsics) {
            return glm::translate(glm::dvec3(camera_translation_and_intrinsics[0],
                                             camera_translation_and_intrinsics[1],
                                             EOS_CERES_USE_PERSPECTIVE ? camera_translation_and_intrinsics[2] : 0.0));
        };

        auto get_projection_matrix(
                const darray<num_cam_params(EOS_CERES_USE_PERSPECTIVE)> &camera_translation_and_intrinsics,
                double aspect) {
            if (EOS_CERES_USE_PERSPECTIVE) {
                const auto &focal = camera_translation_and_intrinsics[3];
                return glm::perspective(focal, aspect, 0.1, 1000.0);
            } else {
                const auto &frustum_scale = camera_translation_and_intrinsics[2];
                return glm::ortho(-1.0 * aspect * frustum_scale, 1.0 * aspect * frustum_scale,
                                  -1.0 * frustum_scale, 1.0 * frustum_scale);
            }
        };

        struct FittingResult {
            explicit FittingResult(
                    const darray<4> &camera_rotation,
                    const darray<num_cam_params(EOS_CERES_USE_PERSPECTIVE)> &camera_translation_and_intrinsics,
                    const glm::dvec4 &viewport,
                    double aspect) : aspect(aspect), viewport(viewport) {
                quaternion_rotation = glm::dquat(camera_rotation[0], camera_rotation[1],
                                                 camera_rotation[2], camera_rotation[3]);
                euler_angles_rotation = glm::eulerAngles(quaternion_rotation);
                rotation_matrix = glm::mat4_cast(quaternion_rotation);

                translation_matrix = ceres_example::get_translation_matrix(camera_translation_and_intrinsics);
                projection_matrix = ceres_example::get_projection_matrix(camera_translation_and_intrinsics, aspect);
            }

            double aspect;
            glm::dquat quaternion_rotation;
            glm::dvec3 euler_angles_rotation;
            glm::dmat4x4 rotation_matrix;
            glm::dmat4x4 translation_matrix;
            glm::dmat4x4 projection_matrix;
            glm::dvec4 viewport;
        };

        template<typename LandmarkType>
        void draw_landmarks(Mat &image, const LandmarkCollection<LandmarkType> &landmarks,
                            const cv::Scalar &color = {0.0, 255.0, 255.0}) {
            for (const auto &landmark : landmarks) {
                cv::rectangle(image, cv::Point2d(landmark.coordinates[0] - 2.0f, landmark.coordinates[1] - 2.0f),
                              cv::Point2d(landmark.coordinates[0] + 2.0f, landmark.coordinates[1] + 2.0f), color);
            }
        }

        template<typename LandmarkType>
        void draw_mesh_vertices(Mat &image, const core::Mesh &mesh,
                                const LandmarkCollection<LandmarkType> &landmarks,
                                const FittingResult &fitting_result,
                                const cv::Scalar &color = {0.0f, 0.0f, 255.0f}) {
            for (const auto &landmark : landmarks) {
                const auto &vertex = mesh.vertices[landmark.index];
                glm::dvec3 point_3d(vertex[0], vertex[1], vertex[2]); // The 3D model point
                glm::dvec3 projected_point = glm::project(point_3d,
                                                          fitting_result.translation_matrix *
                                                          fitting_result.rotation_matrix,
                                                          fitting_result.projection_matrix,
                                                          fitting_result.viewport);
                cv::circle(image, cv::Point2d(projected_point.x, projected_point.y), 3, color); // red
            }
        }

        template<typename LandmarkType>
        void add_contours(LandmarkCollection<LandmarkType> &landmarks,
                          const std::vector<LandmarkType> &image_points_contour,
                          const std::vector<int> &vertex_indices_contour) {
            landmarks.reserve(landmarks.size() + image_points_contour.size());
            for (int i = 0; i < image_points_contour.size(); ++i) {
                core::Landmark<Vector2f> landmark;
                landmark.coordinates = image_points_contour[i];
                landmark.index = vertex_indices_contour[i];

                landmarks.emplace_back(landmark);
            }
        }
    }
}


template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& out, const std::array<T, N>& a) {
    if (!a.empty()) {
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
    try {
        cli_arguments = ceres_example::parse_cli_arguments(argc, argv);
    } catch (const ceres_example::HelpCallException& e) {
        return EXIT_SUCCESS;
    } catch (const std::invalid_argument& e) {
        return EXIT_FAILURE;
    }

    // Read all data from disk
    ceres_example::FittingData fitting_data;
    try {
        fitting_data = ceres_example::read_fitting_data(cli_arguments.blendshapesfile.string(),
                                                        cli_arguments.contourfile.string(),
                                                        cli_arguments.imagefile.string(),
                                                        cli_arguments.landmarksfile.string(),
                                                        cli_arguments.mappingsfile.string(),
                                                        cli_arguments.modelfile.string(),
                                                        cli_arguments.outputfile.string());
    } catch (const std::runtime_error& e) {
        return EXIT_FAILURE;
    }

    const auto& model_contour = fitting_data.model_contour;
    const auto& ibug_contour = fitting_data.ibug_contour;
    const auto& image = fitting_data.image;
    const auto& morphable_model = fitting_data.morphable_model;
    const auto& landmark_mapper = fitting_data.landmark_mapper;
    const auto& blendshapes = fitting_data.blendshapes;

    // These will be the 2D image points and their corresponding 3D vertex id's used for the fitting
    auto& landmarks = fitting_data.landmarks;
    auto indexed_landmarks = landmark_mapper.get_indexed_landmarks(landmarks);

    google::InitGoogleLogging(argv[0]); // Ceres logging initialisation
    std::stringstream fitting_log;

    // Estimate the camera (pose) from the 2D - 3D point correspondences
    auto start = std::chrono::steady_clock::now();

    // Prepare parameters for fitting
    darray<4> camera_rotation {1.0, 0.0, 0.0, 0.0}; // Quaternion, [w x y z].
    auto camera_translation_and_intrinsics = ceres_example::get_camera_translation_and_intrinsics();
    auto shape_coefficients = darray<EOS_CERES_SHAPES_NUM>();
    auto blendshape_coefficients = darray<EOS_CERES_BLENDSHAPES_NUM>();

    // Create problem for only position fitiing
    Problem camera_problem;

    // Add cost function for position fitiing
    fitting::add_camera_cost_function<EOS_CERES_SHAPES_NUM,
                                      EOS_CERES_BLENDSHAPES_NUM,
                                      EOS_CERES_USE_PERSPECTIVE> (camera_problem,
                                                                  camera_rotation, camera_translation_and_intrinsics,
                                                                  shape_coefficients, blendshape_coefficients,
                                                                  indexed_landmarks, morphable_model, blendshapes,
                                                                  image.cols, image.rows);
    // Block face shape fitting
    camera_problem.SetParameterBlockConstant(&shape_coefficients[0]);
    camera_problem.SetParameterBlockConstant(&blendshape_coefficients[0]);
    if (EOS_CERES_USE_PERSPECTIVE) {
        std::vector<int> vec_constant_extrinsic = {3};
        auto subset_parameterization =
                new ceres::SubsetParameterization(num_cam_params(EOS_CERES_USE_PERSPECTIVE), vec_constant_extrinsic);
        camera_problem.SetParameterization(&camera_translation_and_intrinsics[0], subset_parameterization);
    }

    // Get solver options
    auto solver_options = ceres_example::get_solver_options();

    // Fit position
    Solver::Summary solver_summary;
    Solve(solver_options, &camera_problem, &solver_summary);

    // Log fitting report
    std::cout << solver_summary.BriefReport() << std::endl;

    auto end = std::chrono::steady_clock::now();

    // Draw the mean-face landmarks projected using the estimated camera:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:

    auto fitting_result = ceres_example::FittingResult(camera_rotation, camera_translation_and_intrinsics,
                                                       glm::dvec4(0, image.rows, image.cols, -image.rows),
                                                       static_cast<double>(image.cols) / image.rows);

    Mat outimg = image.clone();
    ceres_example::draw_mesh_vertices(outimg, morphable_model.get_mean(), indexed_landmarks, fitting_result);
    ceres_example::draw_landmarks(outimg, indexed_landmarks);

    fitting_log << "Pose fit with mean shape:\tYaw " << glm::degrees(fitting_result.euler_angles_rotation[1])
                << ", Pitch " << glm::degrees(fitting_result.euler_angles_rotation[0]) << ", Roll "
                << glm::degrees(fitting_result.euler_angles_rotation[2])
                << "; t & f: " << camera_translation_and_intrinsics << '\n'
                << "Ceres took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms." << std::endl;

    const auto& outputfile = cli_arguments.outputfile;
    auto new_path = outputfile.parent_path() / fs::path(outputfile.stem().string() + "_first");
    new_path.replace_extension(".png");
    cv::imwrite(new_path.string(), outimg);

    // Contour fitting:
    if (!fitting_data.ibug_contour.empty()) {
        // These are the additional contour-correspondences we're going to find and then use:
        std::vector<Vector2f> image_points_contour; // the 2D landmark points
        std::vector<int> vertex_indices_contour; // their corresponding 3D vertex indices
        // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
        std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
                fitting::get_contour_correspondences(landmarks, fitting_data.ibug_contour,
                                                     fitting_data.model_contour,
                                                     static_cast<float>(
                                                             glm::degrees(fitting_result.euler_angles_rotation[1])),
                                                     morphable_model.get_mean(),
                                                     fitting_result.translation_matrix * fitting_result.rotation_matrix,
                                                     fitting_result.projection_matrix,
                                                     fitting_result.viewport);

        ceres_example::add_contours(indexed_landmarks, image_points_contour, vertex_indices_contour);
        ceres_example::draw_landmarks(outimg, landmarks); // yellow: subset of the detected LMs that we use
                                                                  //         (draw with contour landmarks)
    }

    // Full fitting - Estimate shape and pose, given the previous pose estimate:
    start = std::chrono::steady_clock::now();
    Problem overall_problem;
    fitting::add_camera_cost_function<EOS_CERES_SHAPES_NUM,
                                      EOS_CERES_BLENDSHAPES_NUM,
                                      EOS_CERES_USE_PERSPECTIVE>(overall_problem,
                                                                 camera_rotation,
                                                                 camera_translation_and_intrinsics,
                                                                 shape_coefficients, blendshape_coefficients,
                                                                 indexed_landmarks, morphable_model,
                                                                 blendshapes, image.cols, image.rows);
    if (EOS_CERES_USE_PERSPECTIVE) {
        std::vector<int> vec_constant_extrinsic = {3};
        auto subset_parameterization =
                new ceres::SubsetParameterization(num_cam_params(EOS_CERES_USE_PERSPECTIVE), vec_constant_extrinsic);
        overall_problem.SetParameterization(&camera_translation_and_intrinsics[0], subset_parameterization);
    }

    // Shape prior:
    fitting::add_shape_prior_cost_function<EOS_CERES_SHAPES_NUM>(overall_problem, shape_coefficients);

    // Prior and constraints on blendshapes:
    fitting::add_blendshape_prior_cost_function<EOS_CERES_BLENDSHAPES_NUM>(overall_problem, blendshape_coefficients);

    // Colour model fitting (this needs a Morphable Model with color (albedo) model, see note above main()):
    Eigen::VectorXf color_instance;
    darray<EOS_CERES_COLOR_COEFFS_NUM> color_coefficients;
    if (morphable_model.has_color_model()) {
        // Add a residual for each vertex:
        fitting::add_image_cost_function<EOS_CERES_SHAPES_NUM,
                                         EOS_CERES_BLENDSHAPES_NUM,
                                         EOS_CERES_COLOR_COEFFS_NUM,
                                         EOS_CERES_USE_PERSPECTIVE>(overall_problem,
                                                                    color_coefficients,
                                                                    camera_rotation, camera_translation_and_intrinsics,
                                                                    shape_coefficients, blendshape_coefficients,
                                                                    morphable_model, blendshapes, image);

        fitting::add_image_prior_cost_function<EOS_CERES_COLOR_COEFFS_NUM>(overall_problem, color_coefficients);
        color_instance = morphable_model.get_color_model().draw_sample(color_coefficients);
    } else {
        std::cout << "The MorphableModel used does not contain a color (albedo) model. No ImageCost will be applied."
                  << std::endl;
        color_instance = Eigen::VectorXf();
    }

    // Set different options for the full fitting:
    /*	solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            //solver_options.linear_solver_type = ceres::DENSE_QR;
            //solver_options.minimizer_type = ceres::TRUST_REGION; // default I think
            //solver_options.minimizer_type = ceres::LINE_SEARCH;
            solver_options.num_threads = 8;
            solver_options.num_linear_solver_threads = 8; // only SPARSE_SCHUR can use this
            solver_options.minimizer_progress_to_stdout = true;
            solver_options.max_num_iterations = 100;
            */
    Solve(solver_options, &overall_problem, &solver_summary);
    std::cout << solver_summary.BriefReport() << std::endl;
    end = std::chrono::steady_clock::now();

    // Draw the landmarks projected using all estimated parameters:
    // Construct the rotation & translation (model-view) matrices, projection matrix, and viewport:

    fitting_result = ceres_example::FittingResult(camera_rotation, camera_translation_and_intrinsics,
                                                  glm::dvec4(0, image.rows, image.cols, -image.rows),
                                                  static_cast<double>(image.cols) / image.rows);

    auto blendshape_coefficients_float = std::vector<float>(std::begin(blendshape_coefficients),
                                                            std::end(blendshape_coefficients));
    auto shape_ceres = morphable_model.get_shape_model().draw_sample(shape_coefficients) +
                       to_matrix(fitting_data.blendshapes) *
                       Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients_float.data(),
                                                         blendshape_coefficients_float.size());

    core::Mesh mesh = morphablemodel::sample_to_mesh(
        shape_ceres, color_instance,
        morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());

    ceres_example::draw_mesh_vertices(outimg, morphable_model.get_mean(),
                                      indexed_landmarks, fitting_result,
                                      {0.0f, 76.0f, 255.0f}); // orange

    fitting_log << "Final fit:\t\t\tYaw " << glm::degrees(fitting_result.euler_angles_rotation[1]) << ", Pitch "
                << glm::degrees(fitting_result.euler_angles_rotation[0]) << ", Roll "
                << glm::degrees(fitting_result.euler_angles_rotation[2])
                << "; t & f: " << camera_translation_and_intrinsics << std::endl;
    fitting_log << "Ceres took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms." << std::endl;

    std::cout << fitting_log.str();

    new_path = outputfile;
    new_path.replace_extension(".obj");
    core::write_obj(mesh, new_path.string());

    new_path.replace_extension(".png");
    cv::imwrite(new_path.string(), outimg);

    return EXIT_SUCCESS;
}
