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

#include <cxxopts.hpp>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <filesystem>

using namespace eos;
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
    cxxopts::Options options("generate-obj", "Generates samples from the model and stores them as obj file "
                                             "as well as outputs a frontal rendering of the sample.");
    // clang-format off
    options.add_options()
        ("h,help", "display the help message")
        ("m,model", "an eos .bin Morphable Model file", cxxopts::value<std::string>(), "filename")
        ("shape-coeffs", "optional comma-separated list of shape coefficients. Do not use spaces between values. "
            "E.g.: '--shape-coeffs 0.0,1.5,-1.0'. All coefficients not specified will be set to zero. "
            "If omitted, the mean is used.",
            cxxopts::value<std::vector<float>>())
        ("color-coeffs", "optional comma-separated list of colour coefficients. Do not use spaces between values. "
            "E.g.: '--colour-coeffs 0.0,1.0,-0.5'. All coefficients not specified will be set to zero. "
            "If omitted, the mean is used.",
            cxxopts::value<std::vector<float>>())
        ("o,output", "name of the output obj file (including .obj). Can be a full path.",
            cxxopts::value<std::string>()->default_value("output.obj"), "filename");
    // clang-format on

    using std::filesystem::path;
    path model_file, output_file;
    vector<float> shape_coefficients, color_coefficients;

    try
    {
        const auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return EXIT_SUCCESS;
        }

        model_file = result["model"].as<std::string>();   // required (with default)
        if (result.count("shape-coeffs"))                 // optional
        {
            shape_coefficients = result["shape-coeffs"].as<std::vector<float>>();
        }
        if (result.count("color-coeffs"))                 // optional
        {
            color_coefficients = result["color-coeffs"].as<std::vector<float>>();
        }
        output_file = result["output"].as<std::string>(); // required (with default)
    } catch (const std::exception& e)
    {
        std::cout << "Error while parsing command-line arguments: " << e.what() << std::endl;
        std::cout << "Use --help to display a list of options." << std::endl;
        return EXIT_FAILURE;
    }

    morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(model_file.string());

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

    core::write_obj(sample_mesh, output_file.string());

    const auto perspective = render::perspective(core::radians(60.0f), 512.0f / 512.0f, 0.1f, 500.0f);
    // Could use orthographic projection accordingly:
    // const auto ortho = render::ortho(-130.0f, 130.0f, -130.0f, 130.0f);
    // (and set enable_near_clipping and enable_far_clipping to false, or use the ortho() overload with
    // appropriate near/far values.)
    Eigen::Matrix4f model_view = Eigen::Matrix4f::Identity();
    model_view(2, 3) = -200.0f; // move the model 200 units back along the z-axis

    const core::Image4u rendering =
        render::render(sample_mesh, model_view, perspective, 512, 512, true, true, true);
    std::filesystem::path filename_rendering(output_file);
    filename_rendering.replace_extension(".png");
    cv::imwrite(filename_rendering.string(), core::to_mat(rendering));

    cout << "Wrote the generated obj and a rendering to files with basename " << output_file << "." << endl;

    return EXIT_SUCCESS;
}
