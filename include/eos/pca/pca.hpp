/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/pca/pca.hpp
 *
 * Copyright 2017 Patrik Huber
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

#ifndef PCA_HPP_
#define PCA_HPP_

#include "eos/morphablemodel/PcaModel.hpp"

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

#include <array>
#include <vector>
#include <utility>
#include <cassert>

namespace eos {
    namespace pca {

/** 
 * A flag specifying how to compute the covariance matrix in the PCA.
 */
enum class Covariance {
    AtA, ///< Compute the traditional covariance matrix A^t*A.
    AAt  ///< Use the inner product, A*A^t, for the covariance matrix.
};

/**
 * @brief Compute PCA on a mean-centred data matrix, and return the eigenvectors and respective eigenvalues.
 * 
 * Computes PCA (principal component analysis) on the given mean-centred data matrix. Note that you
 * should subtract the mean from the data beforehand, this function will not do so.
 * The function computes PCA based on an eigendecomposition of the covariance matrix.
 * If the dimensionality of the data is high, the PCA can be computed on the inner-product matrix
 * A*A^t instead of the covariance matrix A^t*A by setting the flag \c Covariance::AAt.
 * 
 * The function returns n-1 eigenvectors and eigenvalues, where n is the number of data samples given.
 * The eigenvectors and eigenvalues are returned in descending order, with the largest (most significant)
 * first.
 *
 * Note: Changing the \p covariance_type may return eigenvectors with different signs, but otherwise equivalent.
 * This is completely fine as the sign of eigenvectors is arbitrary anyway.
 *
 *
 * Developer notes:
 * If you want to avoid a copy: myarray = np.array(source, order='F') (this will change
 * the numpy array to colmajor, so Eigen can directly accept it.
 * There is other ways how to avoid copies:
 * See: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html.
 * http://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html
 * Also it would be nice if the function could accept any Eigen matrix types (e.g. a MatrixXf or MatrixXd).
 *
 * @param[in] data Mean-free data matrix, with each row being a training sample.
 * @param[in] covariance_type Specifies whether PCA is computed on the covariance matrix AtA (default) or the inner-product matrix AAt.
 * @return A pair containing the matrix of eigenvectors and a vector with the respective eigenvalues.
 */
inline std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf> data, Covariance covariance_type = Covariance::AtA)
{
    using Eigen::VectorXf;
    using Eigen::MatrixXf;

    MatrixXf cov;
    if (covariance_type == Covariance::AtA)
    {
	    cov = data.adjoint() * data;
    }
    else if (covariance_type == Covariance::AAt)
    {
	    cov = data * data.adjoint();
    }

    // The covariance is 1/(n-1) * AtA (or AAt), so divide by (num_samples - 1):
    cov /= (data.rows() - 1);

    const Eigen::SelfAdjointEigenSolver<MatrixXf> eig(cov);

    const int num_eigenvectors_to_keep = data.rows() - 1;

    // Select the eigenvectors and eigenvalues that we want to keep, reverse them (from most significant to least):
    // For 'n' data points, we get at most 'n - 1' non-zero eigenvalues.
    VectorXf eigenvalues = eig.eigenvalues().bottomRows(num_eigenvectors_to_keep).reverse(); // eigenvalues() returns a column-vec
    MatrixXf eigenvectors = eig.eigenvectors().rightCols(num_eigenvectors_to_keep).rowwise().reverse();

    if (covariance_type == Covariance::AAt)
    {
	// Bring the AA^t variant in the right form by multiplying with A^t and 1/sqrt(eval):
	// (see e.g. https://math.stackexchange.com/questions/787822/how-do-covariance-matrix-c-aat-justify-the-actual-ata-in-pca)
	// (note the signs might be different from the AtA solution but that's not a problem as the sign of eigenvectors are arbitrary anyway)
	eigenvectors = data.adjoint() * eigenvectors;

        // Multiply each eigenvector (column) with one over the square root of its respective eigenvalue (1/sqrt(eigenvalue(i))):
        // (this is a neat short-hand notation, see https://stackoverflow.com/a/42945996/1345959).
        const VectorXf one_over_sqrt_eigenvalues = eigenvalues.array().rsqrt();
        eigenvectors *= one_over_sqrt_eigenvalues.asDiagonal();

	// Compensate for the covariance division by (n - 1) above:
	eigenvectors /= std::sqrt(data.rows() - 1);
    }

    return { eigenvectors, eigenvalues };
};

/** 
 * @brief Performs PCA and returns \p num_eigenvectors_to_keep eigenvectors and eigenvalues.
 * 
 * See std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf>, Covariance).
 * 
 * \p num_eigenvectors_to_keep needs to be smaller or equal to n-1, where n is number of rows of data (i.e. number of data samples).
 * 
 * @param[in] data Mean-free data matrix, with each row being a training sample.
 * @param[in] num_eigenvectors_to_keep Specifies how many eigenvectors and eigenvalues to keep.
 * @param[in] covariance_type Specifies whether PCA is computed on the covariance matrix AtA (default) or the inner-product matrix AAt.
 * @return A pair containing the matrix of eigenvectors and a vector with the respective eigenvalues.
 */
inline std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf> data, int num_eigenvectors_to_keep, Covariance covariance_type = Covariance::AtA)
{
    using Eigen::VectorXf;
    using Eigen::MatrixXf;

    VectorXf eigenvalues;
    MatrixXf eigenvectors;
    std::tie(eigenvectors, eigenvalues) = pca(data, covariance_type);

    // Reduce the basis and eigenvalues, and return:
    assert(num_eigenvectors_to_keep <= eigenvectors.size());
    return { eigenvectors.leftCols(num_eigenvectors_to_keep), eigenvalues.topRows(num_eigenvectors_to_keep) };
};

/** 
 * @brief Performs PCA and returns the number of eigenvectors and eigenvalues to retain \p variance_to_keep variance of the original data.
 * 
 * See std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf>, Covariance).
 * 
 * \p variance_to_keep needs to be between 0.0 and 1.0.
 * 
 * @param[in] data Mean-free data matrix, with each row being a training sample.
 * @param[in] variance_to_keep Specifies how much of the variance to retain, in percent (between 0 and 1).
 * @param[in] covariance_type Specifies whether PCA is computed on the covariance matrix AtA (default) or the inner-product matrix AAt.
 * @return A pair containing the matrix of eigenvectors and a vector with the respective eigenvalues.
 */
inline std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf> data, float variance_to_keep, Covariance covariance_type = Covariance::AtA)
{
    using Eigen::VectorXf;
    using Eigen::MatrixXf;

    assert(variance_to_keep >= 0.0f && variance_to_keep <= 1.0f);

    VectorXf eigenvalues;
    MatrixXf eigenvectors;
    std::tie(eigenvectors, eigenvalues) = pca(data, covariance_type);

    // Figure out how many coeffs to keep:
    // variance_explained_by_first_comp = eigenval(1)/sum(eigenvalues)
    // variance_explained_by_second_comp = eigenval(2)/sum(eigenvalues), etc.
    auto num_eigenvectors_to_keep = eigenvalues.size(); // In the "worst" case we return all eigenvectors.
    const auto total_sum = eigenvalues.sum();
    float cum_sum = 0.0f;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        cum_sum += eigenvalues(i);
        // If the current variation explained is larger or equal to the amount of variation that
        // the user requested to keep, we're done:
        if (cum_sum / total_sum >= variance_to_keep) {
            num_eigenvectors_to_keep = i + 1;
            break;
        }
    }

    // Reduce the basis and eigenvalues, and return:
    assert(num_eigenvectors_to_keep <= eigenvectors.size());
    return { eigenvectors.leftCols(num_eigenvectors_to_keep), eigenvalues.topRows(num_eigenvectors_to_keep) };
};

/** 
 * @brief Performs PCA on the given data (including subtracting the mean) and returns the built PcaModel.
 * 
 * See std::pair<Eigen::MatrixXf, Eigen::VectorXf> pca(const Eigen::Ref<const Eigen::MatrixXf>, Covariance) for the details on the PCA.
 * 
 * \p data should be a (num_training_samples x num_data_dimensions) matrix, i.e. each row one data instance (e.g. one 3D scan).
 * 
 * @param[in] data Data matrix (orignal, without the mean subtracted), with each row being a training sample.
 * @param[in] triangle_list Triangle list to build the topology of the PcaModel mesh.
 * @param[in] covariance_type Specifies whether PCA is computed on the covariance matrix AtA (default) or the inner-product matrix AAt.
 * @return A PcaModel constructed from the given data.
 */
inline morphablemodel::PcaModel pca(const Eigen::Ref<const Eigen::MatrixXf> data, std::vector<std::array<int, 3>> triangle_list, Covariance covariance_type = Covariance::AtA)
{
    using Eigen::VectorXf;
    using Eigen::MatrixXf;
    
    // Compute the mean and mean-free data matrix:
    // Each row is one instance of data (e.g. a 3D scan)
    const VectorXf mean = data.colwise().mean();
    const MatrixXf meanfree_data = data.rowwise() - mean.transpose();

    // Carry out PCA and return the constructed model:
    VectorXf eigenvalues;
    MatrixXf eigenvectors;
    std::tie(eigenvectors, eigenvalues) = pca(meanfree_data, covariance_type);

    return morphablemodel::PcaModel(mean, eigenvectors, eigenvalues, triangle_list);
};

    } /* namespace pca */
} /* namespace eos */

#endif /* PCA_HPP_ */
