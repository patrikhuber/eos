/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: matlab/+eos/+fitting/private/fitting.cpp
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
#include "mexplus_eigen.hpp"

#include "mexplus.h"

#include "Eigen/Core"

#include "mex.h"
//#include "matrix.h"

#include <string>

using namespace mexplus;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	using std::string;
	// Check for proper number of input and output arguments:
	mexPrintf("nlhs: %d, nrhs: %d\n", nlhs, nrhs);
	if (nrhs != 12) {
		mexErrMsgIdAndTxt("eos:example:nargin", "fit_shape_and_pose requires 12 input arguments.");
	}
	if (nlhs != 2) { // 'nlhs >= 1' means no output argument apparently?
		mexErrMsgIdAndTxt("eos:example:nargout", "fit_shape_and_pose returns two output arguments.");
	}

	InputArguments input(nrhs, prhs, 12);
	auto morphablemodel_file = input.get<string>(0);
	auto blendshapes_file = input.get<string>(1);
	auto landmarks = input.get<Eigen::MatrixXd>(2);
//	auto mm = input.get<string>(0);
//	double vin1 = input.get<double>(0);
	// Matlab stores col-wise in memory - hence the entry of the second row comes first
	//auto vin2 = input.get<vector<double>>(1);
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

/*	auto asdf = input.get<Eigen::MatrixXd>(1);
	std::stringstream ss2;
	ss2 << asdf;
	std::string msg2 = ss2.str();*/

	OutputArguments output(nlhs, plhs, 2);
	output.set(0, landmarks);
	output.set(1, landmarks);

	//double *vin1, *vin2;
	//vin1 = (double*)mxGetPr(prhs[0]);
	//vin2 = (double*)mxGetPr(prhs[1]);
	//mexPrintf("%f, %f\n", vin1, vin2[0]);
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
