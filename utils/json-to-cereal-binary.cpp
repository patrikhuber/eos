/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/json-to-cereal-binary.cpp
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
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"

#include "cereal/archives/json.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::endl;

/**
 * Reads a json Morphable Model or blendshape file and outputs it
 * as an eos .bin file.
 */
int main(int argc, char *argv[])
{
	fs::path jsonfile, outputfile;
	std::string file_type;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("input,i", po::value<fs::path>(&jsonfile)->required(),
				"input json file (model or blendshapes)")
			("type,t", po::value<std::string>(&file_type)->required(),
				"type of the file to convert - 'model' or 'blendshape'")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("converted_model.bin"),
				"output filename for the converted .bin file")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: json-to-cereal-binary [options]" << endl;
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

	if (file_type == "model")
	{
		morphablemodel::MorphableModel morphable_model;
		
		std::ifstream file(jsonfile.string());
		cereal::JSONInputArchive input_archive(file);
		input_archive(morphable_model);

		morphablemodel::save_model(morphable_model, outputfile.string());
	}
	else if (file_type == "blendshape")
	{
		std::vector<morphablemodel::Blendshape> blendshapes;

		std::ifstream file(jsonfile.string());
		cereal::JSONInputArchive input_archive(file);
		input_archive(blendshapes);

		std::ofstream out_file(outputfile.string(), std::ios::binary);
		cereal::BinaryOutputArchive output_archive(out_file);
		output_archive(blendshapes);
	}
	else
	{
		cout << "Type given is neither 'model' nor 'blendshape'." << endl;
		return EXIT_SUCCESS;
	}

	cout << "Saved eos .bin model as " << outputfile.string() << "." << endl;
	return EXIT_SUCCESS;
}
