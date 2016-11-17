/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/edgestruct-csv-to-json.cpp
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
#include "eos/morphablemodel/EdgeTopology.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"

#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::endl;

// Careful, we're loading them from a 1-based indexing file! And store it as 1-based.
// Some values will contain zeros in the first element, which is a special value (= edge at the boundary of the mesh).
std::vector<std::array<int, 2>> read_edgestruct_csv(std::string filename)
{
	using std::getline;
	using std::string;
	std::vector<std::array<int, 2>> vec;

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open file: " + filename));
	}

	string line;
	while (getline(file, line))
	{
		std::vector<string> vals;
		boost::split(vals, line, boost::is_any_of(","), boost::token_compress_on);
		vec.push_back({ std::stoi(vals[0]), std::stoi(vals[1]) });
	}
	return vec;
};

/**
 * Reads two edgestruct CSV files created from Matlab (one with adjacent faces,
 * one with adjacent vertices), converts them to an EdgeTopology struct, and
 * stores that as json file.
 */
int main(int argc, char *argv[])
{
	fs::path input_adj_faces, input_adj_vertices, outputfile;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("faces,f", po::value<fs::path>(&input_adj_faces)->required(),
				"input edgestruct csv file from Matlab with a list of adjacent faces")
			("vertices,v", po::value<fs::path>(&input_adj_vertices)->required(),
				"input edgestruct csv file from Matlab with a list of adjacent vertices")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("converted_edge_topology.json"),
				"output filename for the converted .json file")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: edgestruct-csv-to-json [options]" << endl;
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

	morphablemodel::EdgeTopology edge_info;
	edge_info.adjacent_faces = read_edgestruct_csv(input_adj_faces.string());
	edge_info.adjacent_vertices = read_edgestruct_csv(input_adj_vertices.string());

	morphablemodel::save_edge_topology(edge_info, outputfile.string());

	cout << "Saved EdgeTopology in json file: " << outputfile.string() << "." << endl;
	return EXIT_SUCCESS;
}
