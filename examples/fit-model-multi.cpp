/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
 *
 * Copyright 2016, 2023 Patrik Huber
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
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/multi_image_fitting.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/render/render.hpp"
#include "eos/render/opencv/draw_utils.hpp"
#include "eos/cpp17/optional.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::vector;
using std::string;

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
    // Note: Could make these all std::string, see fit-model.cpp.
    fs::path modelfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfilebase;
    vector<fs::path> imagefiles, landmarksfiles;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
                "a Morphable Model stored as cereal BinaryArchive")
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
                "basename for the output rendering and obj files");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: fit-model-multi [options]" << endl;
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

    if (landmarksfiles.size() != imagefiles.size()) {
        cout << "Number of landmarks files not equal to number of images given: " << landmarksfiles.size()
             << "!=" << imagefiles.size() << endl;
        return EXIT_FAILURE;
    }

    if (landmarksfiles.empty()) {
        cout << "Please give at least 1 image and landmark file." << endl;
        return EXIT_FAILURE;
    }
    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
    vector<Mat> images;
    for (const auto& imagefile : imagefiles){
        images.push_back(cv::imread(imagefile.string()));
    }
    vector<LandmarkCollection<Eigen::Vector2f>> per_frame_landmarks;
    try
    {
        for (const auto& landmarksfile : landmarksfiles)
        {
            per_frame_landmarks.push_back(core::read_pts_landmarks(landmarksfile.string()));
        }
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
    // The landmark mapper is used to map ibug landmark identifiers to vertex ids:
    core::LandmarkMapper landmark_mapper =
        mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(mappingsfile.string());

    // The expression blendshapes:
    vector<morphablemodel::Blendshape> blendshapes =
        morphablemodel::load_blendshapes(blendshapesfile.string());

    // These two are used to fit the front-facing contour to the ibug contour landmarks:
    fitting::ModelContour model_contour =
        contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
    fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

    // The edge topology is used to speed up computation of the occluding face contour fitting:
    morphablemodel::EdgeTopology edge_topology =
        morphablemodel::load_edge_topology(edgetopologyfile.string());

    // Fit the model, get back a mesh and the pose:
    vector<core::Mesh> per_frame_meshes;
    vector<fitting::RenderingParameters> per_frame_rendering_params;
    vector<int> image_widths;
    vector<int> image_heights;
    for (const auto& image : images)
    {
        image_widths.push_back(image.cols);
        image_heights.push_back(image.rows);
    }

    vector<float> pca_shape_coefficients;
    vector<vector<float>> blendshape_coefficients;
    vector<vector<Eigen::Vector2f>> fitted_image_points;

    std::tie(per_frame_meshes, per_frame_rendering_params) = fitting::fit_shape_and_pose(
        morphable_model, blendshapes, per_frame_landmarks, landmark_mapper, image_widths, image_heights, edge_topology,
        ibug_contour, model_contour, 5, cpp17::nullopt, 30.0f, cpp17::nullopt, pca_shape_coefficients,
        blendshape_coefficients, fitted_image_points);

    cout << "Final pca shape coefficients: ";
    for (auto i : pca_shape_coefficients)
    {
        cout << i << ' ';
    }
    cout << endl;

    WeightedIsomapAveraging texturemap_averaging(60.f); // merge all triangles that are facing <60° towards the camera
    Mat merged_texturemap;
    for (unsigned i = 0; i < images.size(); ++i)
    {
        // The 3D head pose can be recovered as follows - the function returns an Eigen::Vector3f with yaw,
        // pitch, and roll angles:
        const float yaw_angle = per_frame_rendering_params[i].get_yaw_pitch_roll()[0];

        // Extract the texture from the image using given mesh and camera parameters:
        // Have to fiddle around with converting between core::Image and cv::Mat
        const core::Image4u texturemap = render::extract_texture(
            per_frame_meshes[i], per_frame_rendering_params[i].get_modelview(),
            per_frame_rendering_params[i].get_projection(), render::ProjectionType::Orthographic,
            core::from_mat_with_alpha(images[i]));

        // Draw the loaded landmarks:
        Mat outimg = images[i].clone();
        for (const auto& lm : per_frame_landmarks[i])
        {
            cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
                          cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
        }

        // Draw the fitted mesh as wireframe, and save the image:
        render::draw_wireframe(outimg, per_frame_meshes[i], per_frame_rendering_params[i].get_modelview(),
                               per_frame_rendering_params[i].get_projection(),
                               fitting::get_opencv_viewport(images[i].cols, images[i].rows));
        fs::path outputfile = outputfilebase;
        outputfile += fs::path(imagefiles[i].stem());
        outputfile += fs::path(".png");
        cv::imwrite(outputfile.string(), outimg);

        // Save frontal rendering with texture:
        Eigen::Matrix4f modelview_frontal = Eigen::Matrix4f::Identity();
        core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
            morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
            morphable_model.get_color_model().get_mean(),
            morphable_model.get_shape_model().get_triangle_list(),
            morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        const core::Image4u frontal_rendering = render::render(
            neutral_expression, modelview_frontal, render::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256,
            render::create_mipmapped_texture(texturemap), true, false, false);
        outputfile.replace_extension(".frontal.png");
        cv::imwrite(outputfile.string(), core::to_mat(frontal_rendering));
        outputfile.replace_extension("");

        cv::Mat texturemap_opencv = core::to_mat(texturemap);

        // And save the texture map:
        outputfile.replace_extension(".texture.png");
        cv::imwrite(outputfile.string(), texturemap_opencv);

        // Merge the texture maps:
        merged_texturemap = texturemap_averaging.add_and_merge(texturemap_opencv);
    }

    // Save the merged texture map:
    fs::path outputfile = outputfilebase;
    outputfile += fs::path("merged.texture.png");
    cv::imwrite(outputfile.string(), merged_texturemap);
    outputfile.replace_extension("");

    // Save the frontal rendering with merged texture:
    Eigen::Matrix4f modelview_frontal = Eigen::Matrix4f::Identity();
    core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
        morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
        morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(),
        morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
    const core::Image4u frontal_rendering = render::render(
        neutral_expression, modelview_frontal, render::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 512, 512,
        render::create_mipmapped_texture(core::from_mat_with_alpha(merged_texturemap)), true, false, false);
    outputfile.replace_extension(".frontal.png");
    cv::imwrite(outputfile.string(), core::to_mat(frontal_rendering));
    outputfile.replace_extension("");

    // Save the mesh as textured obj:
    outputfile.replace_extension(".obj");
    core::write_textured_obj(morphable_model.draw_sample(pca_shape_coefficients, std::vector<float>()),
                             outputfile.string());

    cout << "Finished fitting and wrote result mesh and texture to files with basename " << outputfilebase
         << "." << endl;

    return EXIT_SUCCESS;
}
