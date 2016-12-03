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

#include "mex.h"
//#include "matrix.h"

#include <iostream>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Check for proper number of input and output arguments:
	mexPrintf("nlhs: %d, nrhs: %d\n", nlhs, nrhs);
	if (nrhs != 2) {
		mexErrMsgIdAndTxt("eos:example:nargin", "example requires two input arguments.");
	}
	else if (nlhs >= 1) {
		mexErrMsgIdAndTxt("eos:example:nargout", "example requires no output argument.");
	}

	double *vin1, *vin2;
	vin1 = (double*)mxGetPr(prhs[0]);
	vin2 = (double*)mxGetPr(prhs[1]);
	mexPrintf("%f, %f\n", *vin1, *vin2);
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
