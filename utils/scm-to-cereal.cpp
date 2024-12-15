/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/scm-to-cereal.cpp
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
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/io/cvssp.hpp"

#include <cxxopts.hpp>

#include <string>
#include <iostream>
#include <filesystem>

/**
 * Reads a CVSSP .scm Morphable Model file and converts it
 * to a cereal binary file.
 */
int main(int argc, char* argv[])
{
    cxxopts::Options options("scm-to-cereal",
                             "Convert a CVSSP .scm morphable model file to an eos (cereal) .bin file.");
    // clang-format off
    options.add_options()
        ("h,help", "display the help message")
        ("m,model", "a CVSSP .scm Morphable Model file",
            cxxopts::value<std::string>(), "filename")
        ("t,isomap", "optional text file containing CVSSP texture mapping coordinates",
            cxxopts::value<std::string>(), "filename")
        ("s,shape-only", "save only the shape-model part of the full 3DMM",
            cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("o,output", "output filename for the Morphable Model in cereal binary format",
            cxxopts::value<std::string>()->default_value("converted_model.bin"), "filename");
    // clang-format on

    std::filesystem::path scmmodelfile, outputfile;
    std::optional<std::string> isomapfile;
    bool save_shape_only;

    try
    {
        const auto result = options.parse(argc, argv);
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            return EXIT_SUCCESS;
        }

        scmmodelfile = result["model"].as<std::string>(); // required
        if (result.count("isomap"))                       // optional
        {
            isomapfile = result["isomap"].as<std::string>();
        }
        save_shape_only = result["shape-only"].as<bool>(); // optional
        outputfile = result["output"].as<std::string>();   // required (with default)
    } catch (const std::exception& e)
    {
        std::cout << "Error while parsing command-line arguments: " << e.what() << std::endl;
        std::cout << "Use --help to display a list of options." << std::endl;
        return EXIT_FAILURE;
    }

    using namespace eos;

    // Load the .scm Morphable Model and save it as cereal model:
    morphablemodel::MorphableModel morphable_model =
        morphablemodel::load_scm_model(scmmodelfile.string(), isomapfile);

    if (save_shape_only)
    {
        // Save only the shape model - to generate the public sfm_shape_3448.bin
        const morphablemodel::MorphableModel shape_only_model(morphable_model.get_shape_model(),
                                                              morphablemodel::PcaModel(), std::nullopt,
                                                              morphable_model.get_texture_coordinates());
        morphablemodel::save_model(shape_only_model, outputfile.string());
    } else
    {
        morphablemodel::save_model(morphable_model, outputfile.string());
    }

    std::cout << "Saved converted model as " << outputfile << "." << std::endl;
    return EXIT_SUCCESS;
}
