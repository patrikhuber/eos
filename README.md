# Eos: A lightweight header-only 3D Morphable Model fitting library in modern C++11/14.


Eos is a tiny 3D Morphable Model fitting library that provides just the bare minimum to load a model and perform camera and shape fitting. It's written in modern C++11/14.

At the moment, it mainly provides the following functionality:

* MorphableModel class to represent a 3DMM (using OpenCVs `cv::Mat`)
* Camera estimation, implementation of the _Gold Standard Algorithm_ from Multiple View Geometry, Hartley & Zisserman
* Shape fitting, implementation of the linear shape-to-landmarks fitting of O. Aldrian & W. Smith, Inverse Rendering of Faces with a 3D Morphable Model, PAMI 2013.

## Usage

* Tested with the following compilers: gcc-4.8.2, clang-3.5, Visual Studio 2013
* Needed dependencies: CMake, Boost system filesystem program_options (1.54.0), OpenCV core, imgproc, highgui (2.4.3). Older versions might work as well.

### Build the examples and tests

* copy `initial_cache.cmake.template` to `initial_cache.cmake`, edit the necessary paths
* create a build directory next to the `eos` folder: `mkdir build; cd build`
* `cmake -C ../eos/initial_cache.cmake -G "<your favourite generator>" ../eos -DCMAKE_INSTALL_PREFIX=install/`
* build using your favourite tools, e.g. `make; make install` or open the solution in Visual Studio.


## Sample code

See examples/fit_model.cpp.


## Documentation

The code is fully documented with Doxygen, and an online version of the documentation can be accessed here (todo - add link). Alternatively, an up-to-date version can be built by setting `-DBUILD_DOCUMENTATION=on` and then building the `doc` target.


## License & contributions

This code is licensed under the Apache License, Version 2.0

Contributions are very welcome! (best in the form of pull requests.) Please use Github issues for any bug reports, ideas, and discussions.

If you use this code in your own work, please cite the following paper: _Fitting 3D Morphable Models using Local Features_, P. Huber, Z. Feng, W. Christmas, J. Kittler, M. Rätsch, IEEE International Conference on Image Processing (ICIP) 2015, Québec City, Canada (http://arxiv.org/abs/1503.02330).
