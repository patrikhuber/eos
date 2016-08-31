# eos: A lightweight header-only 3D Morphable Face Model fitting library in modern C++11/14.
[![GitHub release](http://img.shields.io/github/release/patrikhuber/eos.svg?style=flat-square)][release]
[![Apache License 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg?style=flat-square)][license]

[release]: https://github.com/patrikhuber/eos/releases
[license]: https://github.com/patrikhuber/eos/blob/master/LICENSE

eos is a lightweight 3D Morphable Face Model fitting library that provides just the bare minimum to load a model and perform camera and shape fitting. It's written in modern C++11/14.

At the moment, it mainly provides the following functionality:

* MorphableModel class to represent a 3DMM (using OpenCVs `cv::Mat`)
* Our low-resolution, shape-only 3D Morphable Face Model ([share/sfm_shape_3448.bin](https://github.com/patrikhuber/eos/blob/master/share/sfm_shape_3448.bin))
* Camera pose estimation, implementation of:
  * the _Gold Standard Algorithm_ for estimating an affine camera matrix, from Multiple View Geometry, Hartley & Zisserman
  * a non-linear algorithm that directly estimates the pose angles and camera translation
* Shape fitting, implementation of the linear shape-to-landmarks fitting of O. Aldrian & W. Smith, _Inverse Rendering of Faces with a 3D Morphable Model_, PAMI 2013
* Expression fitting, and 6 linear expression blendshapes: anger, disgust, fear, happiness, sadness, surprise.
* Isomap texture extraction to obtain a pose-invariant representation of the face texture.

## Usage

* Tested with the following compilers: >=gcc-4.8.4, >=clang-3.5, Visual Studio 2015
* Needed dependencies for the library: Boost system (>=1.50.0), OpenCV core (>=2.4.3)

To use the library in your own project, just add the following directories to your include path:

* `eos/include`
* `eos/3rdparty/cereal-1.1.1/include`
* `eos/3rdparty/glm-0.9.7.0`

### Build the examples and tests

* Needed dependencies for the example app: CMake (>=2.8.10), Boost system, filesystem, program_options (>=1.50.0), OpenCV core, imgproc, highgui (>=2.4.3).

To build:

```
git clone https://github.com/patrikhuber/eos.git
mkdir build && cd build # creates a build directory next to the _eos_ folder
cmake -G "<your favourite generator>" ../eos -DCMAKE_INSTALL_PREFIX=../install/`
make && make install # or open the project file and build in an IDE like Visual Studio
```
If some dependencies can't be found, copy `initial_cache.cmake.template` to `initial_cache.cmake`, edit the necessary paths and run `cmake` with `-C ../eos/initial_cache.cmake`.


## Sample code

The fit-model example app creates a 3D face from a 2D image.

After `make install` or running the `INSTALL` target, an example image with landmarks can be found in `install/bin/data/`. The model and the necessary landmarks mapping file are installed to `install/share/`.

You can run the example just by running:

`fit-model`

Or, by manually specifying the face model, landmark-to-vertex mappings, an image and its 2D landmarks:

`fit-model -m ../share/sfm_shape_3448.bin -p ../share/ibug2did.txt -i data/image_0010.png -l data/image_0010.pts`

The output is an `obj` file with the shape and a `png` with the extracted isomap. The estimated pose angles and shape coefficients are available in the code via the API.

See [examples/fit-model.cpp](https://github.com/patrikhuber/eos/blob/master/examples/fit-model.cpp) for the full code.

## The Surrey Face Model

The library includes a low-resolution shape-only version of the Surrey Morphable Face Model. It is a PCA model of shape variation built from 3D face scans. It comes with uv-coordinates to perform texture remapping.

<img src="https://raw.githubusercontent.com/patrikhuber/eos/gh-pages/images/sfm_shape_3448_mesh.png" width=20% alt="Face Model Picture"></img>

The full model will be available soon too via academic licencing.

## Documentation

Doxygen: http://patrikhuber.github.io/eos/doc/

The [fit-model example](https://github.com/patrikhuber/eos/blob/master/examples/fit-model.cpp) and the [Namespace List](http://patrikhuber.github.io/eos/doc/namespaces.html) in doxygen are a good place to start.


## License & contributions

This code is licensed under the Apache License, Version 2.0. The 3D morphable face model under [share/sfm_shape_3448.bin](https://github.com/patrikhuber/eos/blob/master/share/sfm_shape_3448.bin) is free for use for non-commercial purposes. For commercial purposes, contact the [Centre for Vision, Speech and Signal Processing](http://www.surrey.ac.uk/cvssp/).

Contributions are very welcome! (best in the form of pull requests.) Please use Github issues for any bug reports, ideas, and discussions.

If you use this code in your own work, please cite the following paper: _A Multiresolution 3D Morphable Face Model and Fitting Framework_, P. Huber, G. Hu, R. Tena, P. Mortazavian, W. Koppen, W. Christmas, M. Rätsch, J. Kittler, International Conference on Computer Vision Theory and Applications (VISAPP) 2016, Rome, Italy [[PDF]](http://www.patrikhuber.ch/files/3DMM_Framework_VISAPP_2016.pdf).
