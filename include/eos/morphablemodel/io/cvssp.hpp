/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/io/cvssp.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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
#pragma once

#ifndef IO_CVSSP_HPP_
#define IO_CVSSP_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"

#include "opencv2/core/core.hpp"

#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace eos {
namespace morphablemodel {

// Forward declaration
std::vector<std::array<double, 2>> load_isomap(std::string isomap_file);

/**
 * Load a shape or color model from a .scm file containing
 * a Morphable Model in the Surrey format. CVSSP's software
 * internally trains and stores the model in this custom binary
 * format and this class provides means to load them.
 *
 * Note on multi-resolution models: The landmarks to vertex-id mapping is
 * always the same. The lowest resolution model has all the landmarks defined
 * and for the higher resolutions, the mesh is divided from that on.
 * Note: For new landmarks we add, this might not be the case if we add them
 * in the highest resolution model, so take care!
 *
 * The PCA basis matrix stored in the file and loaded is the orthogonal PCA basis, i.e. it is not normalised
 * by the eigenvalues.
 *
 * @param[in] model_filename A binary .scm-file containing the model.
 * @param[in] isomap_file An optional path to an isomap containing texture coordinates.
 * @return The Morphable Model loaded from the file.
 * @throws ...
 */
inline MorphableModel load_scm_model(std::string model_filename, std::string isomap_file = std::string())
{
    using cv::Mat;
    if (sizeof(unsigned int) != 4) // note/todo: maybe use uint32 or similar instead? Yep, but still we could encounter endianness-trouble.
    {
        std::cout << "Warning: We're reading 4 Bytes from the file but sizeof(unsigned int) != 4. Check the code/behaviour." << std::endl;
    }
    if (sizeof(double) != 8)
    {
        std::cout << "Warning: We're reading 8 Bytes from the file but sizeof(double) != 8. Check the code/behaviour." << std::endl;
    }

    std::ifstream model_file(model_filename, std::ios::binary);
    if (!model_file)
    {
        const std::string msg("Unable to open model file: " + model_filename);
        std::cout << msg << std::endl;
        throw std::runtime_error(msg);
    }

    // Reading the shape model
    // Read (reference?) num triangles and vertices
    unsigned int num_vertices = 0;
    unsigned int num_triangles = 0;
    model_file.read(reinterpret_cast<char*>(&num_vertices),
                    4); // 1 char = 1 byte. uint32=4bytes. float64=8bytes.
    model_file.read(reinterpret_cast<char*>(&num_triangles), 4);

    // Read triangles
    std::vector<std::array<int, 3>> triangle_list;

    triangle_list.resize(num_triangles);
    unsigned int v0, v1, v2;
    for (unsigned int i = 0; i < num_triangles; ++i)
    {
        v0 = v1 = v2 = 0;
        model_file.read(reinterpret_cast<char*>(&v0), 4); // would be nice to pass a &vector and do it in one
        model_file.read(reinterpret_cast<char*>(&v1), 4); // go, but didn't work. Maybe a cv::Mat would work?
        model_file.read(reinterpret_cast<char*>(&v2), 4);
        triangle_list[i][0] = v0;
        triangle_list[i][1] = v1;
        triangle_list[i][2] = v2;
    }

    // Read number of rows and columns of the shape projection matrix (pcaBasis)
    unsigned int num_shape_pca_coeffs = 0;
    unsigned int num_shape_dims = 0; // dimension of the shape vector (3*num_vertices)
    model_file.read(reinterpret_cast<char*>(&num_shape_pca_coeffs), 4);
    model_file.read(reinterpret_cast<char*>(&num_shape_dims), 4);

    if (3 * num_vertices != num_shape_dims)
    {
        std::cout << "Warning: Number of shape dimensions is not equal to three times the number of "
                     "vertices. Something will probably go wrong during the loading."
                  << std::endl;
    }

    // Read shape projection matrix
    Mat orthonormal_pca_basis_shape(num_shape_dims, num_shape_pca_coeffs,
                                    CV_32FC1); // m x n (rows x cols) = numShapeDims x numShapePcaCoeffs
    std::cout << "Loading shape PCA basis matrix with " << orthonormal_pca_basis_shape.rows << " rows and "
              << orthonormal_pca_basis_shape.cols << " cols." << std::endl;
    for (unsigned int col = 0; col < num_shape_pca_coeffs; ++col)
    {
        for (unsigned int row = 0; row < num_shape_dims; ++row)
        {
            double var = 0.0;
            model_file.read(reinterpret_cast<char*>(&var), 8);
            orthonormal_pca_basis_shape.at<float>(row, col) = static_cast<float>(var);
        }
    }

    // Read mean shape vector
    unsigned int mean_dims = 0; // dimension of the mean (3*num_vertices)
    model_file.read(reinterpret_cast<char*>(&mean_dims), 4);
    if (mean_dims != num_shape_dims)
    {
        std::cout << "Warning: Number of shape dimensions is not equal to the number of dimensions of the "
                     "mean. Something will probably go wrong during the loading."
                  << std::endl;
    }
    Mat mean_shape(mean_dims, 1, CV_32FC1);
    unsigned int counter = 0;
    double vd0, vd1, vd2;
    for (unsigned int i = 0; i < mean_dims / 3; ++i)
    {
        vd0 = vd1 = vd2 = 0.0;
        model_file.read(reinterpret_cast<char*>(&vd0), 8);
        model_file.read(reinterpret_cast<char*>(&vd1), 8);
        model_file.read(reinterpret_cast<char*>(&vd2), 8);
        mean_shape.at<float>(counter, 0) = static_cast<float>(vd0);
        ++counter;
        mean_shape.at<float>(counter, 0) = static_cast<float>(vd1);
        ++counter;
        mean_shape.at<float>(counter, 0) = static_cast<float>(vd2);
        ++counter;
    }

    // Read shape eigenvalues
    unsigned int num_eigenvals_shape = 0;
    model_file.read(reinterpret_cast<char*>(&num_eigenvals_shape), 4);
    if (num_eigenvals_shape != num_shape_pca_coeffs)
    {
        std::cout << "Warning: Number of coefficients in the PCA basis matrix is not equal to the number of "
                     "eigenvalues. Something will probably go wrong during the loading."
                  << std::endl;
    }
    Mat eigenvalues_shape(num_eigenvals_shape, 1, CV_32FC1);
    for (unsigned int i = 0; i < num_eigenvals_shape; ++i)
    {
        double var = 0.0;
        model_file.read(reinterpret_cast<char*>(&var), 8);
        eigenvalues_shape.at<float>(i, 0) = static_cast<float>(var);
    }

    // Todo: We should change these to read into an Eigen matrix directly, and not into a cv::Mat first.
    using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Map<RowMajorMatrixXf> orthonormal_pca_basis_shape_(orthonormal_pca_basis_shape.ptr<float>(),
                                                                    orthonormal_pca_basis_shape.rows,
                                                                    orthonormal_pca_basis_shape.cols);
    const Eigen::Map<RowMajorMatrixXf> eigenvalues_shape_(eigenvalues_shape.ptr<float>(),
                                                          eigenvalues_shape.rows, eigenvalues_shape.cols);
    const Eigen::Map<RowMajorMatrixXf> mean_shape_(mean_shape.ptr<float>(), mean_shape.rows, mean_shape.cols);
    const PcaModel shape_model(mean_shape_, orthonormal_pca_basis_shape_, eigenvalues_shape_, triangle_list);

    // Reading the color model
    // Read number of rows and columns of projection matrix
    unsigned int num_color_pca_coeffs = 0;
    unsigned int num_color_dims = 0;
    model_file.read(reinterpret_cast<char*>(&num_color_pca_coeffs), 4);
    model_file.read(reinterpret_cast<char*>(&num_color_dims), 4);
    // Read color projection matrix
    Mat orthonormal_pca_basis_color(num_color_dims, num_color_pca_coeffs, CV_32FC1);
    std::cout << "Loading color PCA basis matrix with " << orthonormal_pca_basis_color.rows << " rows and "
              << orthonormal_pca_basis_color.cols << " cols." << std::endl;
    for (unsigned int col = 0; col < num_color_pca_coeffs; ++col)
    {
        for (unsigned int row = 0; row < num_color_dims; ++row)
        {
            double var = 0.0;
            model_file.read(reinterpret_cast<char*>(&var), 8);
            orthonormal_pca_basis_color.at<float>(row, col) = static_cast<float>(var);
        }
    }

    // Read mean color vector
    unsigned int color_mean_dims = 0; // dimension of the mean (3*num_vertices)
    model_file.read(reinterpret_cast<char*>(&color_mean_dims), 4);
    Mat mean_color(color_mean_dims, 1, CV_32FC1);
    counter = 0;
    for (unsigned int i = 0; i < color_mean_dims / 3; ++i)
    {
        vd0 = vd1 = vd2 = 0.0;
        model_file.read(reinterpret_cast<char*>(&vd0),
                        8); // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
        model_file.read(reinterpret_cast<char*>(&vd1), 8);
        model_file.read(reinterpret_cast<char*>(&vd2), 8);
        mean_color.at<float>(counter, 0) = static_cast<float>(vd0);
        ++counter;
        mean_color.at<float>(counter, 0) = static_cast<float>(vd1);
        ++counter;
        mean_color.at<float>(counter, 0) = static_cast<float>(vd2);
        ++counter;
    }

    // Read color eigenvalues
    unsigned int num_eigenvals_color = 0;
    model_file.read(reinterpret_cast<char*>(&num_eigenvals_color), 4);
    Mat eigenvalues_color(num_eigenvals_color, 1, CV_32FC1);
    for (unsigned int i = 0; i < num_eigenvals_color; ++i)
    {
        double var = 0.0;
        model_file.read(reinterpret_cast<char*>(&var), 8);
        eigenvalues_color.at<float>(i, 0) = static_cast<float>(var);
    }

    // Todo: We should change these to read into an Eigen matrix directly, and not into a cv::Mat first.
    const Eigen::Map<RowMajorMatrixXf> orthonormal_pca_basis_color_(orthonormal_pca_basis_color.ptr<float>(),
                                                                    orthonormal_pca_basis_color.rows,
                                                                    orthonormal_pca_basis_color.cols);
    const Eigen::Map<RowMajorMatrixXf> eigenvalues_color_(eigenvalues_color.ptr<float>(),
                                                          eigenvalues_color.rows, eigenvalues_color.cols);
    const Eigen::Map<RowMajorMatrixXf> mean_color_(mean_color.ptr<float>(), mean_color.rows, mean_color.cols);
    const PcaModel color_model(mean_color_, orthonormal_pca_basis_color_, eigenvalues_color_, triangle_list);

    model_file.close();

    // Load the isomap with texture coordinates if a filename has been given:
    std::vector<std::array<double, 2>> tex_coords;
    if (!isomap_file.empty())
    {
        tex_coords = load_isomap(isomap_file);
        if (shape_model.get_data_dimension() / 3.0f != tex_coords.size())
        {
            const std::string error_msg("Error, wrong number of texture coordinates. Don't have the same "
                                        "number of texcoords than the shape model has vertices.");
            std::cout << error_msg << std::endl;
            throw std::runtime_error(error_msg);
        }
    }

    return MorphableModel(shape_model, color_model, cpp17::nullopt, tex_coords);
};

/**
 * Load a set of 2D texture coordinates pre-generated by the isomap algorithm.
 * After loading, we rescale the coordinates to [0, 1] x [0, 1].
 *
 * @param[in] isomap_file Path to an isomap file containing texture coordinates.
 * @return The 2D texture coordinates for every vertex.
 * @throws ...
 */
inline std::vector<std::array<double, 2>> load_isomap(std::string isomap_file)
{
    using std::string;
    std::vector<float> x_coords, y_coords;
    string line;
    std::ifstream file(isomap_file);
    if (!file)
    {
        const string error_msg("The isomap file could not be opened. Did you specify a correct filename? " + isomap_file);
        throw std::runtime_error(error_msg);
    } else
    {
        while (getline(file, line))
        {
            std::istringstream iss(line);
            string x, y;
            iss >> x >> y;
            x_coords.push_back(std::stof(x));
            y_coords.push_back(std::stof(y));
        }
        file.close();
    }
    // Process the coordinates: Find the min/max and rescale to [0, 1] x [0, 1]
    const auto min_max_x = std::minmax_element(begin(x_coords), end(x_coords)); // min_max_x is a pair, first=min, second=max
    const auto min_max_y = std::minmax_element(begin(y_coords), end(y_coords));

    std::vector<std::array<double, 2>> tex_coords;
    const float divisor_x = *min_max_x.second - *min_max_x.first;
    const float divisor_y = *min_max_y.second - *min_max_y.first;
    for (int i = 0; i < x_coords.size(); ++i)
    {
        tex_coords.push_back(std::array<double, 2>{
            (x_coords[i] - *min_max_x.first) / divisor_x,
            1.0f - (y_coords[i] - *min_max_y.first) / divisor_y}); // We rescale to [0, 1] and at the same
                                                                   // time flip the y-coords (because in the
                                                                   // isomap, the coordinates are stored
                                                                   // upside-down).
    }

    return tex_coords;
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* IO_CVSSP_HPP_ */
