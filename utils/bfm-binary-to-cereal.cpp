/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/bfm-binary-to-cereal.cpp
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

#include "Eigen/Core"

#include "cxxopts.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace eos;
using std::cout;
using std::endl;
using std::vector;

std::vector<std::array<double, 2>> read_texcoords_from_obj(std::string obj_file)
{
	std::vector<std::array<double, 2>> texcoords;

	std::ifstream file(obj_file);
	if (!file.is_open()) {
		throw std::runtime_error(std::string("Could not open landmark file: " + obj_file));
	}

	std::string line;
	while (getline(file, line))
	{
		std::string first_two = line.substr(0, 2);
		if (first_two != "vt") {
			continue;
		}
		std::stringstream lineStream(line);

		std::array<double, 2> tc;
		std::string throw_away;
		if (!(lineStream >> throw_away >> tc[0] >> tc[1])) {
			throw std::runtime_error(std::string("Texture coordinates format error while parsing the line: " + line));
		}
		texcoords.push_back(tc);
	}
	return texcoords;
}

/**
 * Reads a raw binary file created with share/convert_bfm2009_to_raw_binary.m
 * and outputs it as an eos .bin file. Optionally, an .obj file can be given -
 * the texture coordinates from that obj will then be read and used as the
 * model's texture coordinates (as the BFM comes without texture coordinates).
 */
int main(int argc, char *argv[])
{
	std::string bfm_file, obj_file, outputfile;
	std::string file_type;
	try {
		cxxopts::Options options("bfm-binary-to-cereal");
		options.add_options()
			("h,help", "display the help message")
			("i,input", "input raw binary model file from Matlab script",
				cxxopts::value<std::string>(bfm_file))
			("t,texture-coordinates", "optional .obj file to read texture coordinates from",
				cxxopts::value<std::string>(obj_file)) // optional argument
			("o,output", "output filename for the converted .bin file",
				cxxopts::value<std::string>(outputfile)->default_value("bfm2009.bin"))
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

	std::ifstream file(bfm_file, std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Unable to open model file: " << bfm_file << std::endl;
		return EXIT_FAILURE;
	}

	// We process the texcoords first, as reading the model takes longer
	std::vector<std::array<double, 2>> texture_coordinates;
	if (!obj_file.empty())
	{
		texture_coordinates = read_texcoords_from_obj(obj_file);
	}

	// Read the shape model - first some dimensions:
	int num_vertices = 0;
	{
		int num_vertices_times_three = 0; // the data dimension
		file.read(reinterpret_cast<char*>(&num_vertices_times_three), 4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
		if (num_vertices_times_three % 3 != 0)
		{
			std::cout << "Shape: num_vertices_times_three % 3 != 0" << std::endl;
			return EXIT_FAILURE;
		}
		num_vertices = num_vertices_times_three / 3;
	}
	int num_shape_basis_vectors = 0;
	file.read(reinterpret_cast<char*>(&num_shape_basis_vectors), 4);

	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	// Read the mean:
	VectorXf mean_shape(num_vertices * 3);
	for (int i = 0; i < num_vertices * 3; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		mean_shape(i) = value;
	}

	// Read the orthonormal shape basis matrix:
	MatrixXf orthonormal_pca_basis_shape(num_vertices * 3, num_shape_basis_vectors); // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << "Loading shape PCA basis matrix with " << orthonormal_pca_basis_shape.rows() << " rows and " << orthonormal_pca_basis_shape.cols() << " cols." << std::endl;
	for (int col = 0; col < num_shape_basis_vectors; ++col) {
		for (int row = 0; row < num_vertices * 3; ++row) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), 4);
			orthonormal_pca_basis_shape(row, col) = value;
		}
	}

	// Read the shape eigenvalues:
	VectorXf eigenvalues_shape(num_shape_basis_vectors);
	for (int i = 0; i < num_shape_basis_vectors; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		eigenvalues_shape(i) = value;
	}

	// Read number of triangles and then the triangle list:
	// We additionally subtract 1 to each triangle index, since
	// the BFM triangle indices start at 1, not at 0.
	int num_triangles = 0;
	file.read(reinterpret_cast<char*>(&num_triangles), 4);
	std::vector<std::array<int, 3>> triangle_list;
	triangle_list.resize(num_triangles);
	int v0, v1, v2;
	for (int i = 0; i < num_triangles; ++i) {
		v0 = v1 = v2 = 0;
		file.read(reinterpret_cast<char*>(&v0), 4);	// would be nice to pass a &vector and do it in one
		file.read(reinterpret_cast<char*>(&v1), 4);	// go, but didn't work. Maybe a cv::Mat would work?
		file.read(reinterpret_cast<char*>(&v2), 4);
		triangle_list[i][0] = v0 - 1;
		triangle_list[i][1] = v1 - 1;
		triangle_list[i][2] = v2 - 1;
	}

	morphablemodel::PcaModel shape_model(mean_shape, orthonormal_pca_basis_shape, eigenvalues_shape, triangle_list);

	// Reading the colour (albedo) model:
	int num_vertices_color = 0;
	{
		int num_vertices_times_three = 0; // the data dimension
		file.read(reinterpret_cast<char*>(&num_vertices_times_three), 4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
		if (num_vertices_times_three % 3 != 0)
		{
			std::cout << "Colour: num_vertices_times_three % 3 != 0" << std::endl;
			return EXIT_FAILURE;
		}
		num_vertices_color = num_vertices_times_three / 3;
	}
	int num_color_basis_vectors = 0;
	file.read(reinterpret_cast<char*>(&num_color_basis_vectors), 4);

	// Read the mean:
	// We additionally divide each value by 255 to get from the domain
	// of values in the BFM ([0, 255]) to the SFM ([0, 1]).
	VectorXf mean_color(num_vertices_color * 3);
	for (int i = 0; i < num_vertices_color * 3; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		mean_color(i) = value / 255.0f;
	}

	// Read the orthonormal colour basis matrix:
	MatrixXf orthonormal_pca_basis_color(num_vertices_color * 3, num_color_basis_vectors); // m x n (rows x cols) = num_colour_dims x num_colour_bases
	std::cout << "Loading colour PCA basis matrix with " << orthonormal_pca_basis_color.rows() << " rows and " << orthonormal_pca_basis_color.cols() << " cols." << std::endl;
	for (int col = 0; col < num_color_basis_vectors; ++col) {
		for (int row = 0; row < num_vertices_color * 3; ++row) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), 4);
			orthonormal_pca_basis_color(row, col) = value;
		}
	}

	// Read the colour eigenvalues:
	VectorXf eigenvalues_color(num_color_basis_vectors);
	for (int i = 0; i < num_color_basis_vectors; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		eigenvalues_color(i) = value;
	}

	morphablemodel::PcaModel color_model(mean_color, orthonormal_pca_basis_color, eigenvalues_color, triangle_list);

	file.close();

	if (shape_model.get_data_dimension() / 3 != texture_coordinates.size())
	{
		std::cout << "Warning: PCA shape model's data dimension is different from the number of texture coordinates given. The converted model is still saved, but does most likely not work correctly for texturing." << std::endl;
	}

	morphablemodel::MorphableModel morphable_model(shape_model, color_model, texture_coordinates);
	morphablemodel::save_model(morphable_model, outputfile);

	cout << "Saved eos .bin model as " << outputfile << "." << endl;
	return EXIT_SUCCESS;
}
