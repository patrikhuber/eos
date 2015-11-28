/*
 * Eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/PcaModel.hpp
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

#ifndef PCAMODEL_HPP_
#define PCAMODEL_HPP_

#include "eos/morphablemodel/io/mat_cerealisation.hpp"
#include "cereal/access.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"

#include "opencv2/core/core.hpp"

#include <string>
#include <vector>
#include <array>
#include <random>

namespace eos {
	namespace morphablemodel {

// Forward declarations of free functions
cv::Mat normalise_pca_basis(cv::Mat unnormalised_basis, cv::Mat eigenvalues);
cv::Mat unnormalise_pca_basis(cv::Mat normalised_basis, cv::Mat eigenvalues);

/**
 * @brief This class represents a PCA-model that consists of:
 *   - a mean vector (y x z)
 *   - a PCA basis matrix (unnormalised and normalised)
 *   - a PCA variance vector.
 *
 * It also contains a list of triangles to built a mesh as well as a mapping
 * from landmark points to the corresponding vertex-id in the mesh.
 * It is able to return instances of the model as meshes.
 */
class PcaModel
{
public:
	PcaModel() {}; // workaround for a VS2015 RC bug. Change to '=default' in RTM.

	/**
	 * Construct a PCA model from given mean, normalised PCA basis, eigenvalues
	 * and triangle list.
	 *
	 * See the documentation of the member variables for how the data should
	 * be arranged.
	 *
	 * @param[in] mean The mean used to build the PCA model.
	 * @param[in] pca_basis The PCA basis (eigenvectors), normalised (multiplied by the eigenvalues).
	 * @param[in] eigenvalues The eigenvalues used to build the PCA model.
	 * @param[in] triangle_list An index list of how to assemble the mesh.
	 */
	PcaModel(cv::Mat mean, cv::Mat pca_basis, cv::Mat eigenvalues, std::vector<std::array<int, 3>> triangle_list) : mean(mean), normalised_pca_basis(pca_basis), eigenvalues(eigenvalues), triangle_list(triangle_list)
	{
		const auto seed = std::random_device()();
		engine.seed(seed);
		unnormalised_pca_basis = unnormalise_pca_basis(normalised_pca_basis, eigenvalues);
	};

	/**
	 * Returns the number of principal components in the model.
	 *
	 * @return The number of principal components in the model.
	 */
	int get_num_principal_components() const
	{
		// Note: we could assert(normalised_pca_basis.cols==unnormalised_pca_basis.cols)
		return normalised_pca_basis.cols;
	};

	/**
	 * Returns the dimension of the data, i.e. the number of shape dimensions.
	 *
	 * As the data is arranged in a [x y z x y z ...] fashion, dividing this by
	 * three yields the number of vertices in the model.
	 *
	 * @return The dimension of the data.
	 */
	int get_data_dimension() const
	{
		// Note: we could assert(normalised_pca_basis.rows==unnormalised_pca_basis.rows)
		return normalised_pca_basis.rows;
	};

	/**
	 * Returns a list of triangles on how to assemble the vertices into a mesh.
	 *
	 * @return The list of triangles to build a mesh.
	 */
	std::vector<std::array<int, 3>> get_triangle_list() const
	{
		return triangle_list;
	};

	/**
	 * Returns the mean of the model.
	 *
	 * @return The mean of the model.
	 */
	cv::Mat get_mean() const
	{
		return mean;
	};

	/**
	 * Return the value of the mean at a given vertex index.
	 *
	 * @param[in] vertex_index A vertex index.
	 * @return A homogeneous vector containing the values at the given vertex index.
	 */
	cv::Vec4f get_mean_at_point(int vertex_index) const
	{
		vertex_index *= 3;
		if (vertex_index >= mean.rows) {
			throw std::out_of_range("The given vertex id is larger than the dimension of the mean.");
		}
		return cv::Vec4f(mean.at<float>(vertex_index), mean.at<float>(vertex_index + 1), mean.at<float>(vertex_index + 2), 1.0f);
	};

	/**
	 * Draws a random sample from the model, where the coefficients are drawn
	 * from a standard normal (or with the given standard deviation).
	 *
	 * @param[in] sigma The standard deviation.
	 * @return A random sample from the model.
	 */
	cv::Mat draw_sample(float sigma = 1.0f)
	{
		std::normal_distribution<float> distribution(0.0f, sigma); // this constructor takes the stddev

		std::vector<float> alphas(get_num_principal_components());

		for (auto&& a : alphas) {
			a = distribution(engine);
		}

		return draw_sample(alphas);
	};

	/**
	 * Returns a sample from the model with the given PCA coefficients.
	 * The given coefficients should follow a standard normal distribution, i.e.
	 * not be "normalised" with their eigenvalues/variances.
	 *
	 * @param[in] coefficients The PCA coefficients used to generate the sample.
	 * @return A model instance with given coefficients.
	 */
	cv::Mat draw_sample(std::vector<float> coefficients) const
	{
		// Fill the rest with zeros if not all coefficients are given:
		if (coefficients.size() < get_num_principal_components()) {
			coefficients.resize(get_num_principal_components());
		}
		cv::Mat alphas(coefficients);

		cv::Mat model_sample = mean + normalised_pca_basis * alphas;

		return model_sample;
	};

	/**
	 * Returns the PCA basis matrix, i.e. the eigenvectors.
	 * Each column of the matrix is an eigenvector.
	 * The returned basis is normalised, i.e. every eigenvector
	 * is normalised by multiplying it with the square root of its eigenvalue.
	 *
	 * Returns a clone of the matrix so that the original cannot
	 * be modified. TODO: No, don't return a clone.
	 *
	 * @return Returns the normalised PCA basis matrix.
	 */
	cv::Mat get_normalised_pca_basis() const
	{
		return normalised_pca_basis.clone();
	};

	/**
	 * Returns the PCA basis for a particular vertex.
	 * The returned basis is normalised, i.e. every eigenvector
	 * is normalised by multiplying it with the square root of its eigenvalue.
	 *
	 * @param[in] vertex_id A vertex index. Make sure it is valid.
	 * @return A Mat that points to the rows in the original basis.
	 */
	cv::Mat get_normalised_pca_basis(int vertex_id) const
	{
		vertex_id *= 3; // the basis is stored in the format [x y z x y z ...]
		return normalised_pca_basis.rowRange(vertex_id, vertex_id + 3);
	};

	/**
	 * Returns the PCA basis matrix, i.e. the eigenvectors.
	 * Each column of the matrix is an eigenvector.
	 * The returned basis is unnormalised, i.e. not scaled by their eigenvalues.
	 *
	 * Returns a clone of the matrix so that the original cannot
	 * be modified. TODO: No, don't return a clone.
	 *
	 * @return Returns the unnormalised PCA basis matrix.
	 */
	cv::Mat get_unnormalised_pca_basis() const
	{
		return unnormalised_pca_basis.clone();
	};

	/**
	 * Returns the PCA basis for a particular vertex.
	 * The returned basis is unnormalised, i.e. not scaled by their eigenvalues.
	 *
	 * @param[in] vertex_id A vertex index. Make sure it is valid.
	 * @return A Mat that points to the rows in the original basis.
	 */
	cv::Mat get_unnormalised_pca_basis(int vertex_id) const
	{
		vertex_id *= 3; // the basis is stored in the format [x y z x y z ...]
		return unnormalised_pca_basis.rowRange(vertex_id, vertex_id + 3);
	};

	/**
	 * Returns an eigenvalue.
	 *
	 * @param[in] index The index of the eigenvalue to return.
	 * @return The eigenvalue.
	 */
	float get_eigenvalue(int index) const
	{
		return eigenvalues.at<float>(index);
	};

private:
	std::mt19937 engine; ///< Random number engine used to draw random coefficients.

	cv::Mat mean; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices.
	cv::Mat normalised_pca_basis; ///< The normalised PCA basis matrix. m x n (rows x cols) = numShapeDims x numShapePcaCoeffs, (=eigenvector matrix V). Each column is an eigenvector.
	cv::Mat unnormalised_pca_basis; ///< The unnormalised PCA basis matrix. m x n (rows x cols) = numShapeDims x numShapePcaCoeffs, (=eigenvector matrix V). Each column is an eigenvector.
	cv::Mat eigenvalues; ///< A col-vector of the eigenvalues (variances in the PCA space).

	std::vector<std::array<int, 3>> triangle_list; ///< List of triangles that make up the mesh of the model.

	friend class cereal::access;
	/**
	 * Serialises this class using cereal.
	 *
	 * @param[in] ar The archive to serialise to (or to serialise from).
	 */
	template<class Archive>
	void serialize(Archive& archive)
	{
		archive(mean, normalised_pca_basis, unnormalised_pca_basis, eigenvalues, triangle_list);
		// Note: If the files are too big, We could split this in save/load, only
		// store one of the bases, and calculate the other one when loading.
	};
};


/**
 * Takes an unnormalised PCA basis matrix (a matrix consisting
 * of the eigenvectors and normalises it, i.e. multiplies each
 * eigenvector by the square root of its corresponding
 * eigenvalue.
 *
 * @param[in] unnormalised_basis An unnormalised PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The normalised PCA basis matrix.
 */
inline cv::Mat normalise_pca_basis(cv::Mat unnormalised_basis, cv::Mat eigenvalues)
{
	using cv::Mat;
	Mat normalised_basis(unnormalised_basis.size(), unnormalised_basis.type()); // empty matrix with the same dimensions
	Mat sqrt_of_eigenvalues = eigenvalues.clone();
	for (int i = 0; i < eigenvalues.rows; ++i) {
		sqrt_of_eigenvalues.at<float>(i) = std::sqrt(eigenvalues.at<float>(i));
	}
	// Normalise the basis: We multiply each eigenvector (i.e. each column) with the square root of its corresponding eigenvalue
	for (int basis = 0; basis < unnormalised_basis.cols; ++basis) {
		Mat normalised_eigenvector = unnormalised_basis.col(basis).mul(sqrt_of_eigenvalues.at<float>(basis));
		normalised_eigenvector.copyTo(normalised_basis.col(basis));
	}

	return normalised_basis;
};

/**
 * Takes a normalised PCA basis matrix (a matrix consisting
 * of the eigenvectors and denormalises it, i.e. multiplies each
 * eigenvector by 1 over the square root of its corresponding
 * eigenvalue.
 *
 * @param[in] normalised_basis A normalised PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The unnormalised PCA basis matrix.
 */
inline cv::Mat unnormalise_pca_basis(cv::Mat normalised_basis, cv::Mat eigenvalues)
{
	using cv::Mat;
	Mat unnormalised_basis(normalised_basis.size(), normalised_basis.type()); // empty matrix with the same dimensions
	Mat one_over_sqrt_of_eigenvalues = eigenvalues.clone();
	for (int i = 0; i < eigenvalues.rows; ++i) {
		one_over_sqrt_of_eigenvalues.at<float>(i) = 1.0f / std::sqrt(eigenvalues.at<float>(i));
	}
	// De-normalise the basis: We multiply each eigenvector (i.e. each column) with 1 over the square root of its corresponding eigenvalue
	for (int basis = 0; basis < normalised_basis.cols; ++basis) {
		Mat unnormalised_eigenvector = normalised_basis.col(basis).mul(one_over_sqrt_of_eigenvalues.at<float>(basis));
		unnormalised_eigenvector.copyTo(unnormalised_basis.col(basis));
	}

	return unnormalised_basis;
};

	} /* namespace morphablemodel */
} /* namespace eos */

#endif /* PCAMODEL_HPP_ */
