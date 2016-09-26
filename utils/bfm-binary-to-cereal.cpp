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

#include "opencv2/core/core.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::endl;
using std::vector;
using cv::Mat;

std::vector<cv::Vec2f> read_texcoords_from_obj(std::string obj_file)
{
	std::vector<cv::Vec2f> texcoords;

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

		cv::Vec2f tc;
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
	fs::path bfm_file, obj_file, outputfile;
	std::string file_type;
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
				"display the help message")
			("input,i", po::value<fs::path>(&bfm_file)->required(),
				"input raw binary model file from Matlab script")
			("texture-coordinates,t", po::value<fs::path>(&obj_file),
				"optional .obj file to read texture coordinates from")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("bfm2009.bin"),
				"output filename for the converted .bin file")
			;
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help")) {
			cout << "Usage: bfm-binary-to-cereal [options]" << endl;
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

	std::ifstream file(bfm_file.string(), std::ios::binary);
	if (!file.is_open()) {
		std::cout << "Unable to open model file: " << bfm_file.string() << std::endl;
		return EXIT_FAILURE;
	}

	// We process the texcoords first, as reading the model takes longer
	std::vector<cv::Vec2f> texture_coordinates;
	if (!obj_file.empty())
	{
		texture_coordinates = read_texcoords_from_obj(obj_file.string());
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

	// Read the mean:
	// We additionally divide each coordinate by 1000 to get from the domain
	// of values in the BFM (e.g. -57000) to the BFM (values around e.g. -57).
	Mat mean_shape(num_vertices * 3, 1, CV_32FC1);
	for (int i = 0; i < num_vertices * 3; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		mean_shape.at<float>(i) = value / 1000.0f;
	}

	// Read the unnormalised shape basis matrix:
	Mat unnormalised_pca_basis_shape(num_vertices * 3, num_shape_basis_vectors, CV_32FC1); // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
	std::cout << "Loading shape PCA basis matrix with " << unnormalised_pca_basis_shape.rows << " rows and " << unnormalised_pca_basis_shape.cols << " cols." << std::endl;
	for (int col = 0; col < num_shape_basis_vectors; ++col) {
		for (int row = 0; row < num_vertices * 3; ++row) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), 4);
			unnormalised_pca_basis_shape.at<float>(row, col) = value;
		}
	}

	// Read the shape eigenvalues:
	Mat eigenvalues_shape(num_shape_basis_vectors, 1, CV_32FC1);
	for (int i = 0; i < num_shape_basis_vectors; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		eigenvalues_shape.at<float>(i, 0) = value;
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

	// We read the unnormalised basis from the file. Now let's normalise it and store the normalised basis separately.
	Mat normalised_pca_basis_shape = morphablemodel::normalise_pca_basis(unnormalised_pca_basis_shape, eigenvalues_shape);
	morphablemodel::PcaModel shape_model(mean_shape, normalised_pca_basis_shape, eigenvalues_shape, triangle_list);

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
	// of values in the BFM ([0, 255]) to the BFM ([0, 1]).
	Mat mean_color(num_vertices_color * 3, 1, CV_32FC1);
	for (int i = 0; i < num_vertices_color * 3; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		mean_color.at<float>(i) = value / 255.0f;
	}

	// Read the unnormalised colour basis matrix:
	Mat unnormalised_pca_basis_color(num_vertices_color * 3, num_color_basis_vectors, CV_32FC1); // m x n (rows x cols) = num_colour_dims x num_colour_bases
	std::cout << "Loading colour PCA basis matrix with " << unnormalised_pca_basis_color.rows << " rows and " << unnormalised_pca_basis_color.cols << " cols." << std::endl;
	for (int col = 0; col < num_color_basis_vectors; ++col) {
		for (int row = 0; row < num_vertices_color * 3; ++row) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), 4);
			unnormalised_pca_basis_color.at<float>(row, col) = value;
		}
	}

	// Read the colour eigenvalues:
	Mat eigenvalues_color(num_color_basis_vectors, 1, CV_32FC1);
	for (int i = 0; i < num_color_basis_vectors; ++i) {
		float value = 0.0f;
		file.read(reinterpret_cast<char*>(&value), 4);
		eigenvalues_color.at<float>(i, 0) = value;
	}

	// We read the unnormalised basis from the file. Now let's normalise it and store the normalised basis separately.
	Mat normalised_pca_basis_color = morphablemodel::normalise_pca_basis(unnormalised_pca_basis_color, eigenvalues_color);
	morphablemodel::PcaModel color_model(mean_color, normalised_pca_basis_color, eigenvalues_color, triangle_list);

	file.close();

	if (shape_model.get_data_dimension() / 3 != texture_coordinates.size())
	{
		std::cout << "Warning: PCA shape model's data dimension is different from the number of texture coordinates given. The converted model is still saved, but most likely not work correctly for texturing." << std::endl;
	}

	morphablemodel::MorphableModel morphable_model(shape_model, color_model, texture_coordinates);
	morphablemodel::save_model(morphable_model, outputfile.string());

	cout << "Saved eos .bin model as " << outputfile.string() << "." << endl;
	return EXIT_SUCCESS;
}
