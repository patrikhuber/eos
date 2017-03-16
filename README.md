# eos: A lightweight header-only 3D Morphable Face Model fitting library in modern C++11/14.
[![Latest release](http://img.shields.io/github/release/patrikhuber/eos.svg?style=flat-square)][release]
[![Linux build status of master branch](https://img.shields.io/travis/patrikhuber/eos/master.svg?style=flat-square&label=Linux%3A%20build)][travis]
[![Windows build status of master branch](https://ci.appveyor.com/api/projects/status/gekekpn08cdgqcsk/branch/master?svg=true&passingText=Windows%3A%20build%20passing&failingText=Windows%3A%20build%20failing&pendingText=Windows%3A%20build%20pending)][appveyor]
[![Apache License 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg?style=flat-square)][license]

[release]: https://github.com/patrikhuber/eos/releases
[travis]: https://travis-ci.org/patrikhuber/eos
[appveyor]: https://ci.appveyor.com/project/patrikhuber/eos/branch/master
[license]: https://github.com/patrikhuber/eos/blob/master/LICENSE

eos is a lightweight 3D Morphable Face Model fitting library that provides basic functionality to use face models, as well as camera and shape fitting functionality. It's written in modern C++11/14.

At the moment, it mainly provides the following functionality:

* MorphableModel and PcaModel classes to represent 3DMMs, with basic operations like `draw_sample()`
* Our low-resolution, shape-only 3D Morphable Face Model ([share/sfm_shape_3448.bin](https://github.com/patrikhuber/eos/blob/master/share/sfm_shape_3448.bin))
* Fast, linear pose, shape and expression fitting, edge and contour fitting:
  * Linear scaled orthographic projection camera pose estimation
  * Linear shape-to-landmarks fitting, implementation of O. Aldrian & W. Smith, _Inverse Rendering of Faces with a 3D Morphable Model_, PAMI 2013
  * Expression fitting, and 6 linear expression blendshapes: anger, disgust, fear, happiness, sadness, surprise
  * Edge-fitting, heavily inspired by: A. Bas et al., _Fitting a 3D Morphable Model to Edges: A Comparison Between Hard and Soft Correspondences_, ACCVW 2016
* Isomap texture extraction to obtain a pose-invariant representation of the face texture
* (**New**): Python bindings for parts of the library, and Matlab bindings for the fitting
* (_Experimental_): Non-linear fitting cost functions using Ceres for shape, camera, blendshapes and the colour model (needs Ceres to be installed separately)

## Usage

* Tested with the following compilers: >=gcc-4.9, >=clang-3.5, Visual Studio 2015
* Needed dependencies for the library: Boost system (>=1.50.0), OpenCV core (>=2.4.3)

To use the library in your own project, just add the following directories to your include path:

* `eos/include`
* `eos/3rdparty/cereal/include`
* `eos/3rdparty/glm`
* `eos/3rdparty/nanoflann/include`
* `eos/3rdparty/eigen/Eigen`
* `eos/3rdparty/eigen3-nnls/src`

**Make sure to clone with `--recursive` to download the required submodules!**

### Build the examples and tests

* Needed dependencies for the example app: CMake (>=3.1.3), Boost system, filesystem, program_options (>=1.50.0), OpenCV core, imgproc, highgui (>=2.4.3).

To build:

```
git clone --recursive https://github.com/patrikhuber/eos.git
mkdir build && cd build # creates a build directory next to the 'eos' folder
cmake -G "<your favourite generator>" ../eos -DCMAKE_INSTALL_PREFIX=../install/
make && make install # or open the project file and build in an IDE like Visual Studio
```
If some dependencies can't be found, copy `initial_cache.cmake.template` to `initial_cache.cmake`, edit the necessary paths and run `cmake` with `-C ../eos/initial_cache.cmake`. On Linux, you may also want to set `-DCMAKE_BUILD_TYPE=...` appropriately.


## Sample code

The fit-model example app creates a 3D face from a 2D image.

After `make install` or running the `INSTALL` target, an example image with landmarks can be found in `install/bin/data/`. The model and the necessary landmarks mapping file are installed to `install/share/`.

You can run the example just by running:

`fit-model`

It will load the face model, landmark-to-vertex mappings, blendshapes, and other required files from the `../share/` directory, and run on the example image. It can be run on other images by giving it a `-i` parameter for the image and `-l` for a set of ibug landmarks. The full set of parameters can be viewed by running `fit-model --help`.

If you are just getting started, it is recommended to have a look at `fit-model-simple` too, as it requires much fewer input, and only fits pose and shape, without any blendshapes or edge-fitting. Its full set of arguments is:

`fit-model-simple -m ../share/sfm_shape_3448.bin -p ../share/ibug2did.txt -i data/image_0010.png -l data/image_0010.pts`

The output in both cases is an `obj` file with the shape and a `png` with the extracted isomap. The estimated pose angles and shape coefficients are available in the code via the API.

See [examples/fit-model.cpp](https://github.com/patrikhuber/eos/blob/master/examples/fit-model.cpp) for the full code.

## The Surrey Face Model

The library includes a low-resolution shape-only version of the Surrey Morphable Face Model. It is a PCA model of shape variation built from 3D face scans. It comes with uv-coordinates to perform texture remapping.

<img src="https://raw.githubusercontent.com/patrikhuber/eos/gh-pages/images/sfm_shape_3448_mesh.png" width=20% alt="Face Model Picture"></img>

The full model is available at [http://www.cvssp.org/facemodel](http://www.cvssp.org/facemodel).

## Python bindings

eos includes python bindings for some of its functionality (and more can be added!). An experimental package is on PyPI: Try `pip install eos-py`. You will still need the data files from this repository.
In case of issues, build the bindings manually: Clone the repository and set `-DEOS_GENERATE_PYTHON_BINDINGS=on` when running `cmake` (and optionally set `PYTHON_EXECUTABLE` to point to your python interpreter if it's not found automatically).

After having obtained the bindings, they can be used like any python module:

```
import eos
import numpy as np

model = eos.morphablemodel.load_model("eos/share/sfm_shape_3448.bin")
sample = model.get_shape_model().draw_sample([1.0, -0.5, 0.7])

help(eos) # check the documentation
```

See [`demo.py`](https://github.com/patrikhuber/eos/blob/master/python/demo.py) for an example on how to run the fitting.

## Matlab bindings

_Experimental_: eos includes Matlab bindings for the `fit_shape_and_pose(...)` function, which means the fitting can be run from Matlab. Set `-DEOS_GENERATE_MATLAB_BINDINGS=on` when running `cmake` to build the required mex-file and run the `INSTALL` target to install everything. (Set `Matlab_ROOT_DIR` to point to your Matlab directory if it's not found automatically). More bindings (e.g. the MorphableModel itself) might be added in the future.

Go to the `install/eos/matlab` directory and run [`demo.m`](https://github.com/patrikhuber/eos/blob/master/matlab/demo.m) to see how to run the fitting. The result is a mesh and rendering parameters (pose).

## Documentation

Doxygen: http://patrikhuber.github.io/eos/doc/

The [fit-model example](https://github.com/patrikhuber/eos/blob/master/examples/fit-model.cpp) and the [Namespace List](http://patrikhuber.github.io/eos/doc/namespaces.html) in doxygen are a good place to start.


## License & contributions

This code is licensed under the Apache License, Version 2.0. The 3D morphable face model under [share/sfm_shape_3448.bin](https://github.com/patrikhuber/eos/blob/master/share/sfm_shape_3448.bin) is free for use for non-commercial purposes. For commercial purposes and to obtain other model resolutions, see [http://www.cvssp.org/facemodel](http://www.cvssp.org/facemodel).

Contributions are very welcome! (best in the form of pull requests.) Please use GitHub issues for any bug reports, ideas, and discussions.

If you use this code in your own work, please cite the following paper: _A Multiresolution 3D Morphable Face Model and Fitting Framework_, P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler, International Conference on Computer Vision Theory and Applications (VISAPP) 2016, Rome, Italy [[PDF]](http://www.patrikhuber.ch/files/3DMM_Framework_VISAPP_2016.pdf).
