/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/morphablemodel/PcaModel.hpp
 *
 * Copyright 2014-2017 Patrik Huber
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

#include "cereal/access.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include "eos/morphablemodel/io/eigen_cerealisation.hpp"

#include "Eigen/Core"

#include <array>
#include <cassert>
#include <random>
#include <string>
#include <vector>

namespace eos {
namespace morphablemodel {

// Forward declarations of free functions:
Eigen::MatrixXf rescale_pca_basis(const Eigen::MatrixXf& orthonormal_basis,
                                  const Eigen::VectorXf& eigenvalues);
Eigen::MatrixXf normalise_pca_basis(const Eigen::MatrixXf& rescaled_basis,
                                    const Eigen::VectorXf& eigenvalues);

/**
 * @brief This class represents a PCA-model that consists of:
 *   - a mean vector (y x z)
 *   - a PCA basis matrix (orthonormal and rescaled)
 *   - a PCA variance vector.
 *
 * It also contains a list of triangles to built a mesh as well as a mapping
 * from landmark points to the corresponding vertex-id in the mesh.
 */
class PcaModel
{
public:
    PcaModel() = default;

    /**
     * Construct a PCA model from given mean, normalised PCA basis, eigenvalues
     * and triangle list.
     *
     * See the documentation of the member variables for how the data should
     * be arranged.
     *
     * @param[in] mean The mean used to build the PCA model.
     * @param[in] orthonormal_pca_basis An orthonormal PCA basis (eigenvectors).
     * @param[in] eigenvalues The eigenvalues used to build the PCA model.
     * @param[in] triangle_list An index list of how to assemble the mesh.
     */
    PcaModel(Eigen::VectorXf mean, Eigen::MatrixXf orthonormal_pca_basis, Eigen::VectorXf eigenvalues,
             std::vector<std::array<int, 3>> triangle_list)
        : mean(mean), orthonormal_pca_basis(orthonormal_pca_basis), eigenvalues(eigenvalues),
          triangle_list(triangle_list)
    {
        rescaled_pca_basis = rescale_pca_basis(orthonormal_pca_basis, eigenvalues);
    };

    /**
     * Returns the number of principal components in the model.
     *
     * @return The number of principal components in the model.
     */
    int get_num_principal_components() const
    {
        // Note: we could assert(rescaled_pca_basis.cols==orthonormal_pca_basis.cols)
        return rescaled_pca_basis.cols();
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
        // Note: we could assert(rescaled_pca_basis.rows==orthonormal_pca_basis.rows)
        return rescaled_pca_basis.rows();
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
    const Eigen::VectorXf& get_mean() const
    {
        return mean;
    };

    /**
     * Return the value of the mean at a given vertex index.
     *
     * @param[in] vertex_index A vertex index.
     * @return A 3-dimensional vector containing the values at the given vertex index.
     */
    Eigen::Vector3f get_mean_at_point(int vertex_index) const
    {
        vertex_index *= 3;
        // Note: We could maybe also return a const& to mean.block(...).
        return Eigen::Vector3f(mean(vertex_index), mean(vertex_index + 1), mean(vertex_index + 2));
    };

    /**
     * Draws a random sample from the model, where the coefficients are drawn
     * from a standard normal (or with the given standard deviation).
     *
     * @param[in] engine Random number engine used to draw random coefficients.
     * @param[in] sigma The standard deviation.
     * @return A random sample from the model.
     */
    template <class RNG>
    Eigen::VectorXf draw_sample(RNG& engine, float sigma = 1.0f) const
    {
        std::normal_distribution<float> distribution(0.0f, sigma); // this constructor takes the stddev

        std::vector<float> alphas(get_num_principal_components());

        for (auto&& a : alphas)
        {
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
    Eigen::VectorXf draw_sample(std::vector<float> coefficients) const
    {
        // Fill the rest with zeros if not all coefficients are given:
        if (coefficients.size() < get_num_principal_components())
        {
            coefficients.resize(get_num_principal_components());
        }
        const Eigen::Map<Eigen::VectorXf> alphas(coefficients.data(), coefficients.size());

        Eigen::VectorXf model_sample = mean + rescaled_pca_basis * alphas;

        return model_sample;
    };

    /**
     * @copydoc PcaModel::draw_sample(std::vector<float>) const
     */
    Eigen::VectorXf draw_sample(std::vector<double> coefficients) const
    {
        // We have to convert the vector of doubles to float:
        const std::vector<float> coeffs_float(std::begin(coefficients), std::end(coefficients));
        return draw_sample(coeffs_float);
    };

    /**
     * Returns the PCA basis matrix, i.e. the eigenvectors.
     * Each column of the matrix is an eigenvector.
     * The returned basis is rescaled, i.e. every eigenvector
     * is rescaled by multiplying it with the square root of its eigenvalue.
     *
     * @return Returns the rescaled PCA basis matrix.
     */
    const Eigen::MatrixXf& get_rescaled_pca_basis() const
    {
        return rescaled_pca_basis;
    };

    /**
     * Returns the PCA basis for a particular vertex, from the rescaled basis.
     *
     * Note: The function does not return the i-th basis vector - it returns all basis
     * vectors, but only the block that is relevant for the vertex \p vertex_id.
     *
     * Todo: Can we return a const & view that points into the original data?
     *
     * @param[in] vertex_id A vertex index. Make sure it is valid.
     * @return A 1x3? 3x1? matrix that points to the rows in the original basis.
     */
    Eigen::MatrixXf get_rescaled_pca_basis_at_point(int vertex_id) const
    {
        vertex_id *= 3;                           // the basis is stored in the format [x y z x y z ...]
        assert(vertex_id < get_data_dimension()); // Make sure the given vertex index isn't larger than the
                                                  // number of model vertices.
        return rescaled_pca_basis.block(vertex_id, 0, 3, rescaled_pca_basis.cols());
    };

    /**
     * Returns the PCA basis matrix, i.e. the eigenvectors.
     * Each column of the matrix is an eigenvector.
     * The returned basis is orthonormal, i.e. the eigenvectors
     * are not scaled by their respective eigenvalues.
     *
     * @return Returns the orthonormal PCA basis matrix.
     */
    const Eigen::MatrixXf& get_orthonormal_pca_basis() const
    {
        return orthonormal_pca_basis;
    };

    /**
     * Returns the PCA basis for a particular vertex, from the orthonormal basis.
     *
     * Note: The function does not return the i-th basis vector - it returns all basis
     * vectors, but only the block that is relevant for the vertex \p vertex_id.
     *
     * @param[in] vertex_id A vertex index. Make sure it is valid.
     * @return A matrix that points to the rows in the original basis.
     */
    Eigen::MatrixXf get_orthonormal_pca_basis_at_point(int vertex_id) const
    {
        vertex_id *= 3;                           // the basis is stored in the format [x y z x y z ...]
        assert(vertex_id < get_data_dimension()); // Make sure the given vertex index isn't larger than the
                                                  // number of model vertices.
        return orthonormal_pca_basis.block(vertex_id, 0, 3, orthonormal_pca_basis.cols());
    };

    /**
     * Returns the models eigenvalues.
     *
     * @return The eigenvalues.
     */
    const Eigen::VectorXf& get_eigenvalues() const
    {
        return eigenvalues;
    };

    /**
     * Returns a specific eigenvalue.
     *
     * @param[in] index The index of the eigenvalue to return.
     * @return The eigenvalue.
     */
    float get_eigenvalue(int index) const
    {
        // no assert - Eigen checks access with an assert in debug builds
        return eigenvalues(index);
    };

private:
    Eigen::VectorXf mean; ///< A 3m x 1 col-vector (xyzxyz...)', where m is the number of model-vertices.
    Eigen::MatrixXf orthonormal_pca_basis; ///< m x n (rows x cols) = numShapeDims x numShapePcaCoeffs,
                                           ///< (=eigenvector matrix V). Each column is an eigenvector.
    Eigen::MatrixXf rescaled_pca_basis;    ///< m x n (rows x cols) = numShapeDims x numShapePcaCoeffs,
                                           ///< (=eigenvector matrix V). Each column is an eigenvector.
    Eigen::VectorXf eigenvalues;           ///< A col-vector of the eigenvalues (variances in the PCA space).

    std::vector<std::array<int, 3>> triangle_list; ///< List of triangles that make up the mesh of the model.

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] archive The archive to serialise to (or to serialise from).
     */
    template <class Archive>
    void serialize(Archive& archive)
    {
        archive(CEREAL_NVP(mean), CEREAL_NVP(orthonormal_pca_basis), CEREAL_NVP(eigenvalues),
                CEREAL_NVP(triangle_list));
        // If we're loading, we have to recompute the rescaled basis, so that it's available (we don't
        // store it anymore):
        if (Archive::is_loading::value)
        {
            rescaled_pca_basis = rescale_pca_basis(orthonormal_pca_basis, eigenvalues);
        }
    };
};

/**
 * Takes an orthonormal PCA basis matrix (a matrix consisting
 * of the eigenvectors) and rescales it, i.e. multiplies each
 * eigenvector by the square root of its corresponding
 * eigenvalue.
 *
 * @param[in] orthonormal_basis An orthonormal_basis PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The rescaled PCA basis matrix.
 */
inline Eigen::MatrixXf rescale_pca_basis(const Eigen::MatrixXf& orthonormal_basis,
                                         const Eigen::VectorXf& eigenvalues)
{
    Eigen::MatrixXf rescaled_basis(orthonormal_basis.rows(),
                                   orthonormal_basis.cols()); // empty matrix with the same dimensions
    const Eigen::VectorXf sqrt_of_eigenvalues = eigenvalues.array().sqrt();
    // Rescale the basis: We multiply each eigenvector (i.e. each column) with the square root of its
    // corresponding eigenvalue
    for (int basis = 0; basis < orthonormal_basis.cols(); ++basis)
    {
        rescaled_basis.col(basis) = orthonormal_basis.col(basis) * sqrt_of_eigenvalues(basis);
    }

    return rescaled_basis;
};

/**
 * Takes a rescaled PCA basis matrix and scales it back to
 * an orthonormal basis matrix by multiplying each eigenvector
 * by 1 over the square root of its corresponding eigenvalue.
 *
 * @param[in] rescaled_basis A rescaled PCA basis matrix.
 * @param[in] eigenvalues A row or column vector of eigenvalues.
 * @return The orthonormal PCA basis matrix.
 */
inline Eigen::MatrixXf normalise_pca_basis(const Eigen::MatrixXf& rescaled_basis,
                                           const Eigen::VectorXf& eigenvalues)
{
    Eigen::MatrixXf orthonormal_basis(rescaled_basis.rows(),
                                      rescaled_basis.cols()); // empty matrix with the same dimensions
    const Eigen::VectorXf one_over_sqrt_of_eigenvalues = eigenvalues.array().rsqrt();
    // "Normalise" the basis: We multiply each eigenvector (i.e. each column) with 1 over the square root of
    // its corresponding eigenvalue
    for (int basis = 0; basis < rescaled_basis.cols(); ++basis)
    {
        orthonormal_basis.col(basis) = rescaled_basis.col(basis) * one_over_sqrt_of_eigenvalues(basis);
    }

    return orthonormal_basis;
};

} /* namespace morphablemodel */
} /* namespace eos */

#endif /* PCAMODEL_HPP_ */
