/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit_model.cpp
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
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/AffineCameraEstimation.hpp"
#include "eos/fitting/LinearShapeFitting.hpp"
#include "eos/morphablemodel/io/cvssp.hpp"
#include "eos/render/utils.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
vector<Vec2f> readPtsLandmarks(std::string filename)
{
	using std::getline;
	vector<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Unable to open landmark file: " + filename));
	}
	
	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);
		Vec2f landmark(0.0f, 0.0f);
		if (!(lineStream >> landmark[0] >> landmark[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being 
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark[0] -= 1.0f;
		landmark[1] -= 1.0f;
		landmarks.emplace_back(landmark);
	}
	return landmarks;
};

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 *
 * First, the 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper. Then, an affine camera matrix
 * is estimated, and then, using this camera matrix, the shape is fitted
 * to the landmarks.
 */
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, imagefile, landmarksfile, mappingsfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required(),
				"a CVSSP .scm Morphable Model file")
			("isomap,t", po::value<fs::path>(&isomapfile),
				"optional isomap containing the texture mapping coordinates")
			("image,i", po::value<fs::path>(&imagefile)->required()->default_value("data/image_0001.png"),
				"an input image")
			("landmarks,l", po::value<fs::path>(&landmarksfile)->required()->default_value("data/image_0001.pts"),
				"2D landmarks for the image, in ibug .pts format")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug2did.txt"),
				"landmark identifier to model vertex number mapping")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit_model [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (po::error& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
	Mat image = cv::imread(imagefile.string());
	auto landmarks = readPtsLandmarks(landmarksfile.string());
	morphablemodel::MorphableModel morphableModel = morphablemodel::loadScmModel(modelfile, isomapfile);
	core::LandmarkMapper landmarkMapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// Draw the loaded landmarks:
	Mat outimg = image.clone();
	for (auto&& lm : landmarks) {
		cv::rectangle(outimg, cv::Point2f(lm[0] - 2.0f, lm[1] - 2.0f), cv::Point2f(lm[0] + 2.0f, lm[1] + 2.0f), { 255, 0, 0 });
	}
	
	// Convert the landmarks to clip-space:
	std::transform(begin(landmarks), end(landmarks), begin(landmarks), [&image](const Vec2f& lm) { return render::screenToClipSpace(lm, image.cols, image.rows); });

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> modelPoints; ///< the points in the 3D shape model
	vector<int> vertexIndices; ///< their vertex indices
	vector<Vec2f> imagePoints; ///< the corresponding 2D landmark points

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	int ibugId = 1;
	for (int i = 0; i < landmarks.size(); ++i) {
		try {
			int vertexIdx = boost::lexical_cast<int>(landmarkMapper.convert(std::to_string(ibugId)));
			Vec4f vertex = morphableModel.getShapeModel().getMeanAtPoint(vertexIdx);
			modelPoints.emplace_back(vertex);
			vertexIndices.emplace_back(vertexIdx);
			imagePoints.emplace_back(landmarks[i]);
		}
		catch (const std::out_of_range&) {
			// just continue if the point isn't defined in the mapping
		}
		++ibugId;
	}
	
	// Estimate the camera from the 2D - 3D point correspondences
	Mat affineCam = fitting::estimate_affine_camera(imagePoints, modelPoints);

	// Draw the mean-face landmarks projected using the estimated camera:
	for (auto&& vertex : modelPoints) {
		Vec2f screenPoint = fitting::project_affine(vertex, affineCam, image.cols, image.rows);
		cv::circle(outimg, cv::Point2f(screenPoint), 5.0f, { 0.0f, 255.0f, 0.0f });
	}

	// Estimate the shape coefficients by fitting the shape to the landmarks:
	float lambda = 5.0f; ///< the regularisation parameter
	vector<float> fittedCoeffs = fitting::fit_shape_to_landmarks_linear(morphableModel, affineCam, imagePoints, vertexIndices, lambda);

	// Obtain the full mesh and draw it using the estimated camera:
	render::Mesh mesh = morphableModel.drawSample(fittedCoeffs, vector<float>());
	render::write_obj(mesh, "out.obj"); // save the mesh as obj

	// Draw the projected points again, this time using the fitted model shape:
	for (auto&& idx : vertexIndices) {
		Vec4f modelPoint(mesh.vertices[idx][0], mesh.vertices[idx][1], mesh.vertices[idx][2], mesh.vertices[idx][3]);
		Vec2f screenPoint = fitting::project_affine(modelPoint, affineCam, image.cols, image.rows);
		cv::circle(outimg, cv::Point2f(screenPoint), 3.0f, { 0.0f, 0.0f, 255.0f });
	}

	// Save the output image:
	cv::imwrite("out.png", outimg);
	cout << "Finished fitting and wrote result to out.png." << endl;

	return EXIT_SUCCESS;
}
