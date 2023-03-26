/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/generate-obj.cpp
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
#include "eos/core/Image.hpp"
#include "eos/core/image/opencv_interop.hpp"
#include "eos/core/write_obj.hpp"
#include "eos/core/math.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/render.hpp"
#include "eos/render/matrix_projection.hpp"
#include "eos/cpp17/optional.hpp"

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
using std::string;
using std::vector;

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

    const auto perspective = render::perspective(core::radians(60.0f), 512.0f / 512.0f, 0.1f, 500.0f);
    // Could use orthographic projection accordingly:
    // const auto ortho = render::ortho(-130.0f, 130.0f, -130.0f, 130.0f);
    // (and set enable_near_clipping and enable_far_clipping to false, or use the ortho() overload with
    // appropriate near/far values.)
    Eigen::Matrix4f model_view = Eigen::Matrix4f::Identity();
    model_view(2, 3) = -200.0f; // move the model 200 units back along the z-axis

    const core::Image4u rendering =
        render::render(sample_mesh, model_view, perspective, 512, 512, true, true, true);
    fs::path filename_rendering(output_file);
    filename_rendering.replace_extension(".png");
    cv::imwrite(filename_rendering.string(), core::to_mat(rendering));

    cout << "Wrote the generated obj and a rendering to files with basename " << output_file << "." << endl;

    return EXIT_SUCCESS;
}
