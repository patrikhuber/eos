cmake_minimum_required(VERSION 3.23...3.30)
project(eos VERSION 1.5.0 LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON) # ON by default from CMake 3.26 onwards

# Set the C++ standard to C++17 and required for all the following targets that we define.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # This makes CMake use -std=c++xx instead of -std=gnu++xx

set(CMAKE_VS_JUST_MY_CODE_DEBUGGING ON)

# All the options for building the library. Can be changed on the command-line or in initial_cache.cmake.
message(STATUS "Options:")
option(EOS_BUILD_EXAMPLES "Build the example applications." ON)
message(STATUS "EOS_BUILD_EXAMPLES: ${EOS_BUILD_EXAMPLES}")
option(EOS_BUILD_CERES_EXAMPLE "Build the fit-model-ceres example (requires Ceres)." OFF)
message(STATUS "EOS_BUILD_CERES_EXAMPLE: ${EOS_BUILD_CERES_EXAMPLE}")
option(EOS_BUILD_UTILS "Build utility applications." OFF)
message(STATUS "EOS_BUILD_UTILS: ${EOS_BUILD_UTILS}")
option(EOS_BUILD_DOCUMENTATION "Build the library documentation." OFF)
message(STATUS "EOS_BUILD_DOCUMENTATION: ${EOS_BUILD_DOCUMENTATION}")
option(EOS_GENERATE_PYTHON_BINDINGS "Build python bindings. Requires python to be installed." OFF)
message(STATUS "EOS_GENERATE_PYTHON_BINDINGS: ${EOS_GENERATE_PYTHON_BINDINGS}")
option(EOS_GENERATE_MATLAB_BINDINGS "Build Matlab bindings. Requires Matlab with the compiler installed or the Matlab Compiler Runtime." OFF)
message(STATUS "EOS_GENERATE_MATLAB_BINDINGS: ${EOS_GENERATE_MATLAB_BINDINGS}")

# Build a CPack driven installer package:
include(InstallRequiredSystemLibraries) # This module will include any runtime libraries that are needed by the project for the current platform
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${eos_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${eos_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${eos_VERSION_PATCH}")
include(CPack)

set(eos_3RDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty") # this is used by the CMakeLists.txt files in the subdirectories

set(CMAKE_MODULE_PATH ${eos_3RDPARTY_DIR}/eigen/cmake)
# Use the Eigen included in the submodule by default, but allow the user to override EIGEN3_INCLUDE_DIR:
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR ${eos_3RDPARTY_DIR}/eigen)
endif()
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}, version: ${EIGEN3_VERSION}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")


# Header files of the eos library:
set(eos_HEADERS
  include/eos/core/Landmark.hpp
  include/eos/core/LandmarkMapper.hpp
  include/eos/core/read_pts_landmarks.hpp
  include/eos/core/Image.hpp
  include/eos/core/image/Pixel.hpp
  include/eos/core/image/PixelTraits.hpp
  include/eos/core/image/utils.hpp
  include/eos/core/image/resize.hpp
  include/eos/core/image/opencv_interop.hpp
  include/eos/core/Mesh.hpp
  include/eos/core/read_obj.hpp
  include/eos/core/write_obj.hpp
  include/eos/core/Rect.hpp
  include/eos/core/math.hpp
  include/eos/morphablemodel/PcaModel.hpp
  include/eos/morphablemodel/MorphableModel.hpp
  include/eos/morphablemodel/Blendshape.hpp
  include/eos/morphablemodel/ExpressionModel.hpp
  include/eos/morphablemodel/coefficients.hpp
  include/eos/morphablemodel/EdgeTopology.hpp
  include/eos/morphablemodel/io/cvssp.hpp
  include/eos/morphablemodel/io/eigen_cerealisation.hpp
  include/eos/pca/pca.hpp
  include/eos/fitting/orthographic_camera_estimation_linear.hpp
  include/eos/fitting/nonlinear_camera_estimation.hpp
  include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp
  include/eos/fitting/detail/eigen_quaternion_cerealisation.hpp
  include/eos/fitting/linear_shape_fitting.hpp
  include/eos/fitting/contour_correspondence.hpp
  include/eos/fitting/blendshape_fitting.hpp
  include/eos/fitting/closest_edge_fitting.hpp
  include/eos/fitting/fitting.hpp
  include/eos/fitting/multi_image_fitting.hpp
  include/eos/fitting/ceres_nonlinear.hpp
  include/eos/fitting/RenderingParameters.hpp
  include/eos/fitting/rotation_angles.hpp
  include/eos/fitting/FittingResult.hpp
  include/eos/render/normals.hpp
  include/eos/render/transforms.hpp
  include/eos/render/matrix_projection.hpp
  include/eos/render/ray_triangle_intersect.hpp
  include/eos/render/draw_utils.hpp
  include/eos/render/opencv/draw_utils.hpp
  include/eos/render/render.hpp
  include/eos/render/detail/TriangleToRasterize.hpp
  include/eos/render/detail/plane.hpp
  include/eos/render/detail/utils.hpp
  include/eos/render/ProjectionType.hpp
  include/eos/render/detail/RayDirection.hpp
  include/eos/render/vertex_visibility.hpp
  include/eos/render/texture_extraction.hpp
  include/eos/render/detail/texturing.hpp
  include/eos/render/SoftwareRenderer.hpp
  include/eos/render/VertexShader.hpp
  include/eos/render/Rasterizer.hpp
  include/eos/render/FragmentShader.hpp
  include/eos/render/Texture.hpp
  include/eos/render/detail/Vertex.hpp
  include/eos/video/Keyframe.hpp
  include/eos/video/keyframe_merging.hpp
)

add_library(eos INTERFACE)
target_sources(eos
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${eos_HEADERS}
)
target_compile_features(eos INTERFACE cxx_std_17)

# Add 3rdparty header includes:
target_include_directories(eos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cereal/include>
    $<INSTALL_INTERFACE:3rdparty/cereal/include>
)
target_include_directories(eos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen>
    $<INSTALL_INTERFACE:3rdparty/eigen>
)
target_include_directories(eos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/nanoflann/include>
    $<INSTALL_INTERFACE:3rdparty/nanoflann/include>
)
target_include_directories(eos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen3-nnls/src>
    $<INSTALL_INTERFACE:3rdparty/eigen3-nnls/src>
)
target_include_directories(eos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/toml11>
    $<INSTALL_INTERFACE:3rdparty/toml11>
)

# Install the eos headers:
install(TARGETS eos
    EXPORT eos-export
    FILE_SET HEADERS
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/share/ DESTINATION share) # the model and metadata
# For 3rd party headers, we only copy the headers and licence files:
install(DIRECTORY ${eos_3RDPARTY_DIR}/cereal/ DESTINATION 3rdparty/cereal) # cereal headers
install(DIRECTORY ${eos_3RDPARTY_DIR}/nanoflann/include/ DESTINATION 3rdparty/nanoflann/include) # nanoflann header
install(FILES ${eos_3RDPARTY_DIR}/nanoflann/COPYING DESTINATION 3rdparty/nanoflann/) # nanoflann licence
install(DIRECTORY ${eos_3RDPARTY_DIR}/eigen3-nnls/src/ DESTINATION 3rdparty/eigen3-nnls/src) # eigen3-nnls header
install(FILES ${eos_3RDPARTY_DIR}/eigen3-nnls/README.md DESTINATION 3rdparty/eigen3-nnls/) # eigen3-nnls attribution
install(DIRECTORY ${eos_3RDPARTY_DIR}/toml11/ DESTINATION 3rdparty/toml11) # toml11 headers
# Install Eigen headers if we're using the Eigen submodule included with eos:
if(EIGEN3_INCLUDE_DIR STREQUAL ${eos_3RDPARTY_DIR}/eigen)
  install(DIRECTORY ${eos_3RDPARTY_DIR}/eigen/Eigen/ DESTINATION 3rdparty/eigen/Eigen)
endif()

if(EOS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(EOS_BUILD_UTILS)
  add_subdirectory(utils)
endif()

if(EOS_BUILD_DOCUMENTATION)
  add_subdirectory(doc)
endif()

if(EOS_GENERATE_PYTHON_BINDINGS)
  set(PYBIND11_PATH "${eos_3RDPARTY_DIR}/pybind11")
  set(PYBIND11_FINDPYTHON ON)
  # Note: I don't think this will work well on Multi-Configuration Generators, but there's nothing we can do:
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	set(PYTHON_IS_DEBUG ON)
  endif()
  # If this fails, the repo has probably not been cloned with submodules. Run: git submodule update --init
  add_subdirectory(${PYBIND11_PATH}) # add and initialise pybind11
  add_subdirectory(python) # the actual bindings
  install(DIRECTORY ${eos_3RDPARTY_DIR}/pybind11/include/ DESTINATION 3rdparty/pybind11/include) # pybind11 headers
  install(FILES ${eos_3RDPARTY_DIR}/pybind11/LICENSE DESTINATION 3rdparty/pybind11/) # pybind11 licence
endif()

if(EOS_GENERATE_MATLAB_BINDINGS)
  add_subdirectory(matlab)
  install(DIRECTORY ${eos_3RDPARTY_DIR}/mexplus/include/ DESTINATION 3rdparty/mexplus/include) # mexplus headers
  install(FILES ${eos_3RDPARTY_DIR}/mexplus/LICENSE DESTINATION 3rdparty/mexplus/) # mexplus licence
endif()
