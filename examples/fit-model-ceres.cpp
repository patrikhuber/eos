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
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/ceres_nonlinear.hpp"

#include "glm/glm.hpp"
#include "glm/ext.hpp"

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "ceres/cubic_interpolation.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace eos;
using namespace glm;
using namespace ceres;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

// print a vector:
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	if (!v.empty()) {
		out << '[';
		std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
		out << "\b\b]";
	}
	return out;
};

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
LandmarkCollection<cv::Vec2f> read_pts_landmarks(std::string filename)
{
	using std::getline;
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}

	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	int ibugId = 1;
	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);

		Landmark<Vec2f> landmark;
		landmark.name = std::to_string(ibugId);
		if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark.coordinates[0] -= 1.0f;
		landmark.coordinates[1] -= 1.0f;
		landmarks.emplace_back(landmark);
		++ibugId;
	}
	return landmarks;
};

/**
 * Single and multi-image non-linear model fitting with Ceres example.
 */
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, blendshapesfile, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
			("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
				"file with blendshapes")
			("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0010.png"),
				"an input image")
			("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
				"2D landmarks for the image, in ibug .pts format")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug2did.txt"),
				"landmark identifier to model vertex number mapping")
			("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
				"file with model contour indices")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
				"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model-ceres [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	google::InitGoogleLogging(argv[0]); // Ceres logging initialisation

	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());

	fitting::ContourLandmarks ibug_contour;
	try {
		ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error reading the contour mappings file: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	Mat image = cv::imread(imagefile.string());
	LandmarkCollection<cv::Vec2f> landmarks;
	try {
		landmarks = read_pts_landmarks(landmarksfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error reading the landmarks: " << e.what() << endl;
		return EXIT_FAILURE;
	}
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	}
	catch (const std::runtime_error& e) {
		cout << "Error loading the Morphable Model: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Note: Actually it's a required argument, so it'll never be empty.
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);
	
	std::vector<eos::morphablemodel::Blendshape> blendshapes = eos::morphablemodel::load_blendshapes(blendshapesfile.string());

	// Draw the loaded landmarks:
	Mat outimg = image.clone();
	for (auto&& lm : landmarks) {
		cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
	}

	constexpr bool use_perspective = false;

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (int i = 0; i < landmarks.size(); ++i) {
		auto converted_name = landmark_mapper.convert(landmarks[i].name);
		if (!converted_name) { // no mapping defined for the current landmark
			continue;
		}
		int vertex_idx = std::stoi(converted_name.get());
		Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
		model_points.emplace_back(vertex);
		vertex_indices.emplace_back(vertex_idx);
		image_points.emplace_back(landmarks[i].coordinates);
	}

	// Estimate the camera (pose) from the 2D - 3D point correspondences
	std::vector<double> camera_rotation; // Quaternion, [w x y z]. Todo: Actually, use std::array for all of these.
	camera_rotation.resize(4); // initialises with zeros
	camera_rotation[0] = 1.0;
	std::vector<double> camera_translation_and_intrinsics;
	constexpr int num_cam_trans_intr_params = use_perspective ? 4 : 3;
	// Parameters for the orthographic projection: [t_x, t_y, frustum_scale]
	// And perspective projection: [t_x, t_y, t_z, fov].
	// Origin is assumed at center of image, and no lens distortions.
	// Note: Actually, we estimate the model-view matrix and not the camera position. But one defines the other.
	camera_translation_and_intrinsics.resize(num_cam_trans_intr_params); // initialises with zeros
	if (use_perspective)
	{
		camera_translation_and_intrinsics[2] = -400.0; // Move the model back (along the -z axis)
		camera_translation_and_intrinsics[3] = glm::radians(45.0f); // fov
	}
	else {
		camera_translation_and_intrinsics[2] = 110.0; // frustum_scale
	}

	std::vector<double> shape_coefficients;
	shape_coefficients.resize(10); // Todo: Currently, the value '10' is hard-coded everywhere. Make it dynamic.
	std::vector<double> blendshape_coefficients;
	blendshape_coefficients.resize(6);

	Problem camera_costfunction;
	for (int i = 0; i < model_points.size(); ++i)
	{
		CostFunction* cost_function = new AutoDiffCostFunction<fitting::LandmarkCost, 2 /* num residuals */, 4 /* camera rotation (quaternion) */, num_cam_trans_intr_params /* camera translation & fov/frustum_scale */, 10 /* shape-coeffs */, 6 /* bs-coeffs */>(new fitting::LandmarkCost(morphable_model.get_shape_model(), blendshapes, image_points[i], vertex_indices[i], image.cols, image.rows, use_perspective));
		camera_costfunction.AddResidualBlock(cost_function, NULL, &camera_rotation[0], &camera_translation_and_intrinsics[0], &shape_coefficients[0], &blendshape_coefficients[0]);
	}
	camera_costfunction.SetParameterBlockConstant(&shape_coefficients[0]); // keep the shape constant
	camera_costfunction.SetParameterBlockConstant(&blendshape_coefficients[0]);
	if (use_perspective)
	{
		camera_costfunction.SetParameterUpperBound(&camera_translation_and_intrinsics[0], 2, -std::numeric_limits<double>::epsilon()); // t_z has to be negative
		camera_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 3, 0.01); // fov in radians, must be > 0
	}
	else {
		camera_costfunction.SetParameterLowerBound(&camera_translation_and_intrinsics[0], 2, 1.0); // frustum_scale must be > 0
	}
	QuaternionParameterization* quaternion_parameterisation = new QuaternionParameterization;
	camera_costfunction.SetParameterization(&camera_rotation[0], quaternion_parameterisation);
	
	Solver::Options solver_options;
	Solver::Summary solver_summary;
	Solve(solver_options, &camera_costfunction, &solver_summary);
	std::cout << solver_summary.BriefReport() << "\n";

	auto mean_mesh = morphable_model.get_mean();
	for (auto&& idx : vertex_indices)
	{
		glm::dvec3 point_3d(mean_mesh.vertices[idx][0], mean_mesh.vertices[idx][1], mean_mesh.vertices[idx][2]); // The 3D model point
		glm::dvec3 projected_point;

		glm::dquat q(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3]);
		auto rot_mtx = glm::mat4_cast(q);
		glm::dvec4 viewport(0, image.rows, image.cols, -image.rows); // OpenCV convention
		const double aspect = static_cast<double>(image.cols) / image.rows;
		
		if (use_perspective)
		{
			auto t_mtx = glm::translate(glm::dvec3(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], camera_translation_and_intrinsics[2]));
			const auto& focal = camera_translation_and_intrinsics[3];
			auto persp_mtx = glm::perspective(focal, aspect, 0.1, 1000.0);
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, persp_mtx, viewport);
		}
		else {
			const auto& frustum_scale = camera_translation_and_intrinsics[2];
			auto t_mtx = glm::translate(glm::dvec3(camera_translation_and_intrinsics[0], camera_translation_and_intrinsics[1], 0.0));
			auto ortho_mtx = glm::ortho(-1.0 * aspect * frustum_scale, 1.0 * aspect * frustum_scale, -1.0 * frustum_scale, 1.0 * frustum_scale);
			projected_point = glm::project(point_3d, t_mtx * rot_mtx, ortho_mtx, viewport);
		}
		cv::circle(outimg, cv::Point2f(projected_point.x, projected_point.y), 3, { 255.0f, 0.0f, 127.0f }); // violet
	}
	for (auto&& lm : image_points) {
		cv::circle(outimg, cv::Point2f(lm), 3, { 255.0f, 0.0f, 0.0f }); // blue - used landmarks
	}

	auto euler_angles = glm::eulerAngles(glm::dquat(camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3])); // returns [P, Y, R]
	std::cout << "Pose fit with mean shape: Yaw " << glm::degrees(euler_angles[1]) << ", Pitch " << glm::degrees(euler_angles[0]) << ", Roll " << glm::degrees(euler_angles[2]) << "; t & f: " << camera_translation_and_intrinsics << '\n';

	return EXIT_SUCCESS;
}
