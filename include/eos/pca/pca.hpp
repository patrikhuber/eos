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
//#include "Eigen/Dense"
#include "Eigen/Eigenvalues"

#include <array>
#include <vector>
#include <utility>

namespace eos {
	namespace pca {

enum class Covariance {
	AtA, // compute the traditional covariance matrix A^t*A.
	AAt  // use the inner product, A*A^t, for the covariance matrix.
};

/**
 * @brief Todo.
 *
 * Todo.
 * If inner_prod=True, then use A*A^t for the covariance instead of A^t * A. The eigenvectors will be computed accordingly so they are identical to the ones computed by A^t * A.
 * Evals and Evecs will be returned in descending order, largest first.
 * Each row in 'data' is a training sample.
 *
 * Note: Changing covariance_type may return eigenvectors with different signs, but otherwise equivalent. This is completely fine, the sign is arbitrary anyway.
 *
 * If you want to avoid a copy: myarray = np.array(source, order='F') (this will change
 * the numpy array to colmajor, so Eigen can directly accept it.
 * There is other ways how to avoid copies:
 * See: https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html.
 * http://pybind11.readthedocs.io/en/master/advanced/cast/eigen.html
 * Also it would be nice if it could accept any Eigen matrix types (e.g. a MatrixXf or MatrixXd).
 *
 * @param[in] data Mean-free!.
 * @param[in] covariance_type Y.
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

	Eigen::SelfAdjointEigenSolver<MatrixXf> eig(cov);

	int num_eigenvectors_to_keep = data.rows() - 1;

	// Select the eigenvectors and eigenvalues that we want to keep, reverse them (from most significant to least):
	// For 'n' data points, do we get 'n' and the last will be zeros/garbage? So we have to remove at least one, always?
	VectorXf z_evals = eig.eigenvalues().bottomRows(num_eigenvectors_to_keep).reverse(); // eigenvalues() returns a column-vec
	MatrixXf z_evecs = eig.eigenvectors().rightCols(num_eigenvectors_to_keep).rowwise().reverse();

	if (covariance_type == Covariance::AAt)
	{
		// Bring the AA^t variant in the right form by multiplying with A^t and 1/sqrt(eval):
		// (see e.g. https://math.stackexchange.com/questions/787822/how-do-covariance-matrix-c-aat-justify-the-actual-ata-in-pca)
		// (note the signs might be different from the AtA solution but that's not a problem as the sign of eigenvectors are arbitrary anyway)
		z_evecs = data.adjoint() * z_evecs;
		for (int c = 0; c < z_evecs.cols(); ++c)
		{
			z_evecs.col(c) *= 1.0 / std::sqrt(z_evals(c));
		}
		// Maybe we can do this instead of the for loop?:
		//VectorXd evals_rsqrts = e_aat.array().rsqrt();
		//RowMajorMatrixXd b_aat_to_ata_all_normalised = b_aat_to_ata_all.cw * evals_rsqrts;
		// What about next 2 lines, not sure:
		//b_aat_to_ata_all.col(2) *= 1.0 / std::sqrt(e_aat(2)); // first eigenvector is identical
		//b_aat_to_ata_all.col(1) *= 1.0 / std::sqrt(e_aat(1)); // this one needs multiplication by -1... odd!

		// Compensate for the covariance divide by (n - 1) above:
		z_evecs /= std::sqrt(data.rows() - 1);
	}

	return { z_evecs, z_evals };
};

inline void pca(int num_coeffs_to_keep)
{
	// Call above pca() function

	// Reduce the basis etc, and return.
};

inline void pca(float variance_to_keep)
{
	// Call above pca() function

	// Figure out how many coeffs to keep:
	// variance_explained_by_first_comp = eigenval(1)/sum(eigenvalues)
	// variance_explained_by_second_comp = eigenval(2)/sum(eigenvalues)
	// Etc...

	// Call pca(...num_coeffs_to_keep...)
	// Return.
};

// data should be MatrixXf data(num_scans, num_data_points), i.e. each row one data instance (e.g. one 3D scan)
inline morphablemodel::PcaModel pca(Eigen::MatrixXf data, std::vector<std::array<int, 3>> triangle_list, Covariance covariance_flag)
{
	using Eigen::VectorXf;
	using Eigen::MatrixXf;
	// Check my Python implementation whether each row is a scan makes sense.
	// Also what's AtA vs AAt called... one is covariance matrix, the other is...?
	// Each row will be a scan:

	VectorXf mean = data.colwise().mean();
	MatrixXf meanfree_data = data.rowwise() - mean.transpose();
	
	
	// Call above PCA functions.
	


	morphablemodel::PcaModel model; // (mean.cast<float>(), z_evecs.cast<float>(), z_evals.cast<float>(), reference_trianglelist);

	return model;
};

	} /* namespace pca */
} /* namespace eos */

#endif /* PCA_HPP_ */
