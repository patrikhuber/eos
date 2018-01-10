/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
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
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/render/render.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
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
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] colour Colour of the mesh to be drawn.
 */
void draw_wireframe(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
{
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
		const auto p2 = glm::project({ mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2] }, modelview, projection, viewport);
		const auto p3 = glm::project({ mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2] }, modelview, projection, viewport);
		if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
		{
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), colour);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), colour);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), colour);
		}
	}
};

/**
 * @brief Merges isomaps from a live video with a weighted averaging, based
 * on the view angle of each vertex to the camera.
 *
 * An optional merge_threshold can be specified upon construction. Pixels with
 * a view-angle above that threshold will be completely discarded. All pixels
 * below the threshold are merged with a weighting based on its vertex view-angle.
 * Assumes the isomaps to be 512x512.
 */
class WeightedIsomapAveraging
{
public:
    /**
     * @brief Constructs a new object that will hold the current averaged isomap and
     * be able to add frames from a live video and merge them on-the-fly.
     *
     * The threshold means: Each triangle with a view angle smaller than the given angle will be used to merge.
     * The default threshold (90°) means all triangles, as long as they're a little bit visible, are merged.
     *
     * @param[in] merge_threshold View-angle merge threshold, in degrees, from 0 to 90.
     */
    WeightedIsomapAveraging(float merge_threshold = 90.0f)
    {
        assert(merge_threshold >= 0.f && merge_threshold <= 90.f);

        visibility_counter = cv::Mat::zeros(512, 512, CV_32SC1);
        merged_isomap = cv::Mat::zeros(512, 512, CV_32FC4);

        // map 0° to 255, 90° to 0:
        float alpha_thresh = (-255.f / 90.f) * merge_threshold + 255.f;
        if (alpha_thresh < 0.f) // could maybe happen due to float inaccuracies / rounding?
            alpha_thresh = 0.0f;
        threshold = static_cast<unsigned char>(alpha_thresh);
    };

    /**
     * @brief Merges the given new isomap with all previously processed isomaps.
     *
     * @param[in] isomap The new isomap to add.
     * @return The merged isomap of all images processed so far, as 8UC4.
     */
    cv::Mat add_and_merge(const cv::Mat& isomap)
    {
        // Merge isomaps, add the current to the already merged, pixel by pixel:
        for (int r = 0; r < isomap.rows; ++r)
        {
            for (int c = 0; c < isomap.cols; ++c)
            {
                if (isomap.at<cv::Vec4b>(r, c)[3] <= threshold)
                {
                    continue; // ignore this pixel, not visible in the extracted isomap of this current frame
                }
                // we're sure to have a visible pixel, merge it:
                // merged_pixel = (old_average * visible_count + new_pixel) / (visible_count + 1)
                merged_isomap.at<cv::Vec4f>(r, c)[0] = (merged_isomap.at<cv::Vec4f>(r, c)[0] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[0]) / (visibility_counter.at<int>(r, c) + 1);
                merged_isomap.at<cv::Vec4f>(r, c)[1] = (merged_isomap.at<cv::Vec4f>(r, c)[1] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[1]) / (visibility_counter.at<int>(r, c) + 1);
                merged_isomap.at<cv::Vec4f>(r, c)[2] = (merged_isomap.at<cv::Vec4f>(r, c)[2] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[2]) / (visibility_counter.at<int>(r, c) + 1);
                merged_isomap.at<cv::Vec4f>(r, c)[3] = 255; // as soon as we've seen the pixel visible once, we set it to visible.
                ++visibility_counter.at<int>(r, c);
            }
        }
        cv::Mat merged_isomap_uchar;
        merged_isomap.convertTo(merged_isomap_uchar, CV_8UC4);
        return merged_isomap_uchar;
    };

private:
    cv::Mat visibility_counter;
    cv::Mat merged_isomap;
    unsigned char threshold;
};

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 * In addition to fit-model-simple, this example uses blendshapes, contour-
 * fitting, and can iterate the fitting.
 *
 * 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper.
 */
int main(int argc, char *argv[])
{
    fs::path modelfile, isomapfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfilebase;
    vector<fs::path> imagefiles, landmarksfiles;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				"a Morphable Model stored as cereal BinaryArchive")
            //("image,i", po::value<vector<fs::path>>(&imagefiles)->required()->default_value("data/image_0010.png"),
            ("image,i", po::value<vector<fs::path>>(&imagefiles)->multitoken(),
				"an input image")
            ("landmarks,l", po::value<vector<fs::path>>(&landmarksfiles)->multitoken(),
				"2D landmarks for the image, in ibug .pts format")
            ("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
				"landmark identifier to model vertex number mapping")
			("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
				"file with model contour indices")
			("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value("../share/sfm_3448_edge_topology.json"),
				"file with model's precomputed edge topology")
			("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value("../share/expression_blendshapes_3448.bin"),
				"file with blendshapes")
            ("output,o", po::value<fs::path>(&outputfilebase)->required()->default_value("out"),
				"basename for the output rendering and obj files")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: fit-model [options]" << endl;
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

    if (landmarksfiles.size() != imagefiles.size()) {
        cout << "Number of landmarksfiles not equal to number of images given: "<<landmarksfiles.size() <<"!=" <<imagefiles.size()<< endl;
        return EXIT_SUCCESS;
    }

    if (landmarksfiles.empty()) {
        cout << "Please give at least 1 image and landmarkfile" << endl;
        return EXIT_SUCCESS;
    }
	// Load the image, landmarks, LandmarkMapper and the Morphable Model:
    vector<Mat> images;
    for (auto& imagefile : imagefiles){
        images.push_back(cv::imread(imagefile.string()));
    }
    vector<LandmarkCollection<cv::Vec2f>> landmarkss;
	try {
        for (auto& landmarksfile : landmarksfiles){
            landmarkss.push_back(read_pts_landmarks(landmarksfile.string()));
        }
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
	// The landmark mapper is used to map ibug landmark identifiers to vertex ids:
	core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile);

	// The expression blendshapes:
	vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	// The edge topology is used to speed up computation of the occluding face contour fitting:
	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// Fit the model, get back a mesh and the pose:
    vector<core::Mesh> meshs;
    vector<fitting::RenderingParameters> rendering_paramss;
    vector<int> image_widths;
    vector<int> image_heights;
    for (auto& image : images) {
        image_widths.push_back(image.cols);
        image_heights.push_back(image.rows);
    }

    std::vector<float> pca_shape_coefficients;
    std::vector<std::vector<float>> blendshape_coefficients;
    std::vector<std::vector<cv::Vec2f>> fitted_image_points;

    std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi(morphable_model, blendshapes, landmarkss, landmark_mapper, image_widths, image_heights, edge_topology, ibug_contour, model_contour, 50, boost::none, 30.0f, boost::none, pca_shape_coefficients, blendshape_coefficients, fitted_image_points);
    //std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi(morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda, boost::none, pca_shape_coefficients, blendshape_coefficients, fitted_image_points);
    //fit_shape_and_pose_multi(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<core::LandmarkCollection<cv::Vec2f>>& landmarks, const core::LandmarkMapper& landmark_mapper, std::vector<int> image_width, std::vector<int> image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations, boost::optional<int> num_shape_coefficients_to_fit, float lambda, boost::optional<fitting::RenderingParameters> initial_rendering_params, std::vector<float>& pca_shape_coefficients, std::vector<std::vector<float>>& blendshape_coefficients, std::vector<std::vector<cv::Vec2f>>& fitted_image_points)

    std::cout<<"final pca shape coefficients: ";
    for (auto i: pca_shape_coefficients)
      std::cout << i << ' ';
    std::cout << std::endl;

    WeightedIsomapAveraging isomap_averaging(60.f); // merge all triangles that are facing <60° towards the camera
    Mat merged_isomap;

    for (unsigned i =0; i <images.size(); ++i) {

        // The 3D head pose can be recovered as follows:
        float yaw_angle = glm::degrees(glm::yaw(rendering_paramss[i].get_rotation()));
        // and similarly for pitch and roll.

        // Extract the texture from the image using given mesh and camera parameters:
        Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_paramss[i], images[i].cols, images[i].rows);
        Mat isomap = render::extract_texture(meshs[i], affine_from_ortho, images[i]);

        // Draw the loaded landmarks:
        Mat outimg = images[i].clone();
        for (auto&& lm : landmarkss[i]) {
            cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f), cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), { 255, 0, 0 });
        }

        // Draw the fitted mesh as wireframe, and save the image:
        draw_wireframe(outimg, meshs[i], rendering_paramss[i].get_modelview(), rendering_paramss[i].get_projection(), fitting::get_opencv_viewport(images[i].cols, images[i].rows));
        fs::path outputfile = outputfilebase;
        outputfile += fs::path(imagefiles[i].stem());
        outputfile += fs::path(".png");
        cv::imwrite(outputfile.string(), outimg);

        //save frontal rendering with texture:
        Mat frontal_rendering;
        glm::mat4 modelview_frontal = glm::mat4( 1.0 );
        core::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(pca_shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        std::tie(frontal_rendering, std::ignore) = render::render(neutral_expression, modelview_frontal, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, render::create_mipmapped_texture(isomap), true, false, false);
        outputfile.replace_extension(".frontal.png");
        cv::imwrite(outputfile.string(), frontal_rendering);
        outputfile.replace_extension("");

        // And save the isomap:
        outputfile.replace_extension(".isomap.png");
        cv::imwrite(outputfile.string(), isomap);

        // merge the isomaps:
        merged_isomap = isomap_averaging.add_and_merge(isomap);
    }

    // save the merged isomap:
    fs::path outputfile = outputfilebase;
    outputfile +=fs::path("merged.isomap.png");
    cv::imwrite(outputfile.string(), merged_isomap);
    outputfile.replace_extension("");

    // save the frontal rendering with merged isomap:
    Mat frontal_rendering;
    glm::mat4 modelview_frontal = glm::mat4( 1.0 );
    core::Mesh neutral_expression = morphablemodel::sample_to_mesh(morphable_model.get_shape_model().draw_sample(pca_shape_coefficients), morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
    std::tie(frontal_rendering, std::ignore) = render::render(neutral_expression, modelview_frontal, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 512, 512, render::create_mipmapped_texture(merged_isomap), true, false, false);
    outputfile.replace_extension(".frontal.png");
    cv::imwrite(outputfile.string(), frontal_rendering);
    outputfile.replace_extension("");

    // Save the mesh as textured obj:
    outputfile.replace_extension(".obj");
    core::write_textured_obj(morphable_model.draw_sample(pca_shape_coefficients,std::vector<float>()), outputfile.string());

    cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfilebase << "." << endl;

	return EXIT_SUCCESS;
}
