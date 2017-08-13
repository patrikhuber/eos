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

#include "cxxopts.hpp"

#include "boost/algorithm/string.hpp"

#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace eos;
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
	std::string input_adj_faces, input_adj_vertices, outputfile;
	try {
		cxxopts::Options options("edgestruct-csv-to-json");
		options.add_options()
			("help,h", "display the help message")
			("faces,f", "input edgestruct csv file from Matlab with a list of adjacent faces",
				cxxopts::value<std::string>(input_adj_faces))
			("vertices,v", "input edgestruct csv file from Matlab with a list of adjacent vertices",
				cxxopts::value<std::string>(input_adj_vertices))
			("output,o", "output filename for the converted .json file",
				cxxopts::value<std::string>(outputfile)->default_value("converted_edge_topology.json"))
			;
		options.parse(argc, argv);
		if (options.count("help")) {
			cout << options.help() << endl;
			return EXIT_SUCCESS;
		}
	}
	catch (const cxxopts::OptionException& e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_FAILURE;
	}

	morphablemodel::EdgeTopology edge_info;
	edge_info.adjacent_faces = read_edgestruct_csv(input_adj_faces);
	edge_info.adjacent_vertices = read_edgestruct_csv(input_adj_vertices);

	morphablemodel::save_edge_topology(edge_info, outputfile);

	cout << "Saved EdgeTopology in json file: " << outputfile << "." << endl;
	return EXIT_SUCCESS;
}
