/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/private/test.cpp
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
#include "mexplus.h"

#include "Eigen/Core"

#include "mex.h"
//#include "matrix.h"

#include <iostream>

namespace mexplus {
// Define template specialisations for Eigen::MatrixXd:
template<>
mxArray* MxArray::from(const Eigen::MatrixXd& eigen_matrix) {
	const int num_rows = static_cast<int>(eigen_matrix.rows());
	const int num_cols = static_cast<int>(eigen_matrix.cols());
	MxArray out_array(MxArray::Numeric<double>(num_rows, num_cols));

	// This might not copy the data but it's evil and probably really dangerous!!!:
	//mxSetData(const_cast<mxArray*>(matrix.get()), (void*)value.data());

	// This copies the data. But I suppose it makes sense that we copy the data when we go
	// from C++ to Matlab, since Matlab can unload the C++ mex module at any time I think.
	// Loop is column-wise
	for (int c = 0; c < num_cols; ++c) {
		for (int r = 0; r < num_rows; ++r) {
			out_array.set(r, c, eigen_matrix(r, c));
		}
	}
	return out_array.release();
};

template<>
void MxArray::to(const mxArray* in_array, Eigen::MatrixXd* eigen_matrix)
{
	MxArray array(in_array);

	if (array.dimensionSize() > 2)
	{
		mexErrMsgIdAndTxt("eos:matlab", "Given array has > 2 dimensions. Can only create 2-dimensional matrices (and vectors).");
	}

	if (array.dimensionSize() == 1 || array.dimensionSize() == 0)
	{
		mexErrMsgIdAndTxt("eos:matlab", "Given array has 0 or 1 dimensions but we expected a 2-dimensional matrix (or row/column vector).");
		// Even when given a single value dimensionSize() is 2, with n=m=1. When does this happen?
	}

	if (!array.isDouble())
	{
		mexErrMsgIdAndTxt("eos:matlab", "Trying to create a Eigen::MatrixXd in C++, but the given data is not of type double.");
	}

	// We can be sure now that the array is 2-dimensional (or 0, but then we're screwed anyway)
	auto nrows = array.dimensions()[0]; // or use array.rows()
	auto ncols = array.dimensions()[1];

	// I think I can just use Eigen::Matrix, not a Map - the Matrix c'tor that we call creates a Map anyway?
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eigen_map(array.getData<double>(), nrows, ncols);
	// Not sure that's alright - who owns the data? I think as it is now, everything points to the data in the mxArray owned by Matlab, but I'm not 100% sure.
	*eigen_matrix = eigen_map;
};

} // namespace mexplus


using namespace mexplus;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	using std::vector;
	// Check for proper number of input and output arguments:
	mexPrintf("nlhs: %d, nrhs: %d\n", nlhs, nrhs);
	if (nrhs != 2) {
		mexErrMsgIdAndTxt("eos:example:nargin", "Example requires two input arguments.");
	}
	else if (nlhs >= 2) { // 'nlhs >= 1' means no output argument apparently?
		mexErrMsgIdAndTxt("eos:example:nargout", "Example requires zero or one output arguments.");
	}

	InputArguments input(nrhs, prhs, 2);
	double vin1 = input.get<double>(0);
	// Matlab stores col-wise in memory - hence the entry of the second row comes first
	auto vin2 = input.get<vector<double>>(1);
/*	auto test = input[1];
	MxArray mxa(test);
	auto ndim = mxa.dimensionSize();
	auto nrows = mxa.dimensions()[0];
	auto ncols = mxa.dimensions()[1];
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> em(vin2.data(), 2, 3);
	// ==> Yes, now I can put exactly this in the MxArray namespace!
	std::stringstream ss;
	ss << em;
	std::string msg = ss.str();
*/
	//auto x = MxArray::Numeric<double>(2, 2);

	auto asdf = input.get<Eigen::MatrixXd>(1);
	std::stringstream ss2;
	ss2 << asdf;
	std::string msg2 = ss2.str();

	OutputArguments output(nlhs, plhs, 1);
	output.set(0, asdf);

	//double *vin1, *vin2;
	//vin1 = (double*)mxGetPr(prhs[0]);
	//vin2 = (double*)mxGetPr(prhs[1]);
	mexPrintf("%f, %f\n", vin1, vin2[0]);
};

void func()
{
	int x = 4;
};

int func1()
{
	return 5;
};

class MyClass
{
public:
	MyClass() = default;
	int test() {
		return 6;
	};
};
