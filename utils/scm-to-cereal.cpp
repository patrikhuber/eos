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
#include "eos/cpp17/optional.hpp"

#include "boost/program_options.hpp"

#include <string>
#include <iostream>

using namespace eos;
namespace po = boost::program_options;
using std::cout;
using std::endl;

/**
 * Reads a CVSSP .scm Morphable Model file and converts it
 * to a cereal binary file.
 */
int main(int argc, char* argv[])
{
    std::string scmmodelfile, isomapfile, outputfile;
    bool save_shape_only;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<std::string>(&scmmodelfile)->required(),
                "a CVSSP .scm Morphable Model file")
            ("isomap,t", po::value<std::string>(&isomapfile),
                "optional text file containing CVSSP texture mapping coordinates")
            ("shape-only,s", po::value<bool>(&save_shape_only)->default_value(false)->implicit_value(true),
                "save only the shape-model part of the full 3DMM")
            ("output,o", po::value<std::string>(&outputfile)->required()->default_value("converted_model.bin"),
                "output filename for the Morphable Model in cereal binary format");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: scm-to-cereal [options]" << endl;
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

    cpp17::optional<std::string> isomapfile_optional =
        isomapfile.empty() ? cpp17::nullopt : cpp17::optional<std::string>(isomapfile);

    // Load the .scm Morphable Model and save it as cereal model:
    morphablemodel::MorphableModel morphable_model =
        morphablemodel::load_scm_model(scmmodelfile, isomapfile_optional);

    if (save_shape_only)
    {
        // Save only the shape model - to generate the public sfm_shape_3448.bin
        const morphablemodel::MorphableModel shape_only_model(morphable_model.get_shape_model(),
                                                              morphablemodel::PcaModel(), cpp17::nullopt,
                                                              morphable_model.get_texture_coordinates());
        morphablemodel::save_model(shape_only_model, outputfile);
    } else
    {
        morphablemodel::save_model(morphable_model, outputfile);
    }

    cout << "Saved converted model as " << outputfile << "." << endl;
    return EXIT_SUCCESS;
}
