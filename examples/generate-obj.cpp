/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/generate-obj.cpp
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
#include "eos/core/Image.hpp"
#include "eos/core/Image_opencv_interop.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/render.hpp"

#include "glm/gtc/matrix_transform.hpp"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
 * This app generates random samples from the model and stores them as obj file
 * as well as outputs a frontal rendering of the sample.
 *
 * A list of shape and/or colour coefficients can be specified. Any coefficient
 * not specified will be set to zero.
 */
int main(int argc, char* argv[])
{
    string model_file, output_file;
    vector<float> shape_coefficients, color_coefficients;

    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("model", po::value<string>(&model_file)->required(), "an eos .bin Morphable Model file")
            ("shape-coeffs", po::value<vector<float>>(&shape_coefficients)->multitoken(),
                "optional parameter list of shape coefficients. All not specified will be set to zero. E.g.: "
                "'--shape-coeffs 0.0 1.5'. If omitted, the mean is used.")
            ("color-coeffs", po::value<vector<float>>(&color_coefficients)->multitoken(),
                "optional parameter list of colour coefficients. All not specified will be set to zero. E.g.: "
                "'--colour-coeffs 0.0 1.5'. If omitted, the mean is used.")
            ("output", po::value<string>(&output_file)->default_value("output.obj"),
                "name of the output obj file (including .obj). Can be a full path.");
        // clang-format on
        po::variables_map vm;
        // disabling short options to allow negative values for the coefficients, e.g. '--shape-coeffs 0.0 -1.5'
        po::store(
            po::parse_command_line(argc, argv, desc,
                                   po::command_line_style::unix_style ^ po::command_line_style::allow_short),
            vm);
        if (vm.count("help"))
        {
            cout << "Usage: generate-obj [options]" << endl;
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

    morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(model_file);

    if (shape_coefficients.size() < morphable_model.get_shape_model().get_num_principal_components())
    {
        shape_coefficients.resize(morphable_model.get_shape_model().get_num_principal_components());
    }

    if (color_coefficients.size() < morphable_model.get_color_model().get_num_principal_components())
    {
        color_coefficients.resize(morphable_model.get_color_model().get_num_principal_components());
    }

    const core::Mesh sample_mesh = morphable_model.draw_sample(
        shape_coefficients, color_coefficients); // if one of the two vectors is empty, it uses get_mean()

    core::write_obj(sample_mesh, output_file);
    core::Image4u rendering;
    std::tie(rendering, std::ignore) =
        render::render(sample_mesh, glm::mat4x4(1.0f), glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 512, 512,
                       std::nullopt, true, false, false);
    fs::path filename_rendering(output_file);
    filename_rendering.replace_extension(".png");
    cv::imwrite(filename_rendering.string(), core::to_mat(rendering));

    cout << "Wrote the generated obj and a rendering to files with basename " << output_file << "." << endl;

    return EXIT_SUCCESS;
}
