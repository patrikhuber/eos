cmake_minimum_required(VERSION 3.1.3)
project(eos)
set(eos_VERSION_MAJOR 0)
set(eos_VERSION_MINOR 11)
set(eos_VERSION_PATCH 0)
set(eos_VERSION ${eos_VERSION_MAJOR}.${eos_VERSION_MINOR}.${eos_VERSION_PATCH})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# This sets the C++ standard to c++14 and required for all the following targets that we define.
# It has no effect on MSVC though - we thus define more specific requirements for each executable target respectively.
# Also it will not apply to the eos library target, since it is an INTERFACE_LIBRARY, and these properties do not apply to interface libraries.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # This makes CMake use -std=c++11 instead of -std=gnu++11
# This list is likely not complete, but it should be sufficient to error out on old compilers that we cannot build on:
set(EOS_CXX_COMPILE_FEATURES cxx_defaulted_functions cxx_generalized_initializers cxx_generic_lambdas cxx_lambdas cxx_nonstatic_member_init cxx_range_for cxx_right_angle_brackets cxx_strong_enums)

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

# Find dependencies:
find_package(OpenCV REQUIRED core)

if(MSVC)
  # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
  add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
  add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system REQUIRED)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

set(eos_3RDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty") # this is used by the CMakeLists.txt files in the subdirectories

set(CMAKE_MODULE_PATH ${eos_3RDPARTY_DIR}/eigen/cmake)
# Use the Eigen included in the submodule by default, but allow the user to override EIGEN3_INCLUDE_DIR:
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR ${eos_3RDPARTY_DIR}/eigen)
endif()
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}, version: ${EIGEN3_VERSION}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")

# Set the include directories of the 3rd-party submodules that we use:
set(CEREAL_INCLUDE_DIR "${eos_3RDPARTY_DIR}/cereal/include")
set(glm_INCLUDE_DIR "${eos_3RDPARTY_DIR}/glm")
set(nanoflann_INCLUDE_DIR "${eos_3RDPARTY_DIR}/nanoflann/include")
set(eigen3_nnls_INCLUDE_DIR "${eos_3RDPARTY_DIR}/eigen3-nnls/src")

# Header files of the eos library:
set(HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/core/Landmark.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/core/LandmarkMapper.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/core/Mesh.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/PcaModel.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/MorphableModel.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/Blendshape.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/coefficients.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/EdgeTopology.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/io/cvssp.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/morphablemodel/io/mat_cerealisation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/affine_camera_estimation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/orthographic_camera_estimation_linear.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/nonlinear_camera_estimation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/detail/optional_cerealisation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/detail/glm_cerealisation.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/linear_shape_fitting.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/contour_correspondence.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/blendshape_fitting.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/closest_edge_fitting.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/fitting.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/ceres_nonlinear.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/fitting/RenderingParameters.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/render.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/render_affine.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/detail/render_detail.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/detail/render_affine_detail.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/texture_extraction.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/detail/texture_extraction_detail.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/SoftwareRenderer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/VertexShader.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/Rasterizer.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/FragmentShader.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/render/detail/Vertex.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/eos/video/Keyframe.hpp
)

add_library(eos INTERFACE)
target_compile_features(eos INTERFACE ${EOS_CXX_COMPILE_FEATURES})
# Add header includes:
target_include_directories(eos INTERFACE "include")
target_include_directories(eos INTERFACE ${CEREAL_INCLUDE_DIR})
target_include_directories(eos INTERFACE ${Boost_INCLUDE_DIRS})
target_include_directories(eos INTERFACE ${OpenCV_INCLUDE_DIRS})
target_include_directories(eos INTERFACE ${EIGEN3_INCLUDE_DIR})
target_include_directories(eos INTERFACE ${glm_INCLUDE_DIR})
target_include_directories(eos INTERFACE ${nanoflann_INCLUDE_DIR})
target_include_directories(eos INTERFACE ${eigen3_nnls_INCLUDE_DIR})

# Custom target for the library, to make the headers show up in IDEs:
add_custom_target(eos-headers SOURCES ${HEADERS})
source_group(core REGULAR_EXPRESSION include/eos/core/*)
source_group(morphablemodel REGULAR_EXPRESSION include/eos/morphablemodel/*)
source_group(morphablemodel\\io REGULAR_EXPRESSION include/eos/morphablemodel/io/*)
source_group(fitting REGULAR_EXPRESSION include/eos/fitting/*)
source_group(fitting\\detail REGULAR_EXPRESSION include/eos/fitting/detail/*)
source_group(render REGULAR_EXPRESSION include/eos/render/*)
source_group(render\\detail REGULAR_EXPRESSION include/eos/render/detail/*)
source_group(video REGULAR_EXPRESSION include/eos/video/*)

# The eos install target:
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include) # our library headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/share/ DESTINATION share) # the model and metadata
# For 3rd party headers, we only copy the headers and licence files:
install(DIRECTORY ${eos_3RDPARTY_DIR}/cereal/ DESTINATION 3rdparty/cereal) # cereal headers
install(DIRECTORY ${eos_3RDPARTY_DIR}/glm/glm/ DESTINATION 3rdparty/glm/glm) # glm headers
install(FILES ${eos_3RDPARTY_DIR}/glm/copying.txt DESTINATION 3rdparty/glm/) # glm licence
install(DIRECTORY ${eos_3RDPARTY_DIR}/nanoflann/include/ DESTINATION 3rdparty/nanoflann/include) # nanoflann header
install(FILES ${eos_3RDPARTY_DIR}/nanoflann/COPYING DESTINATION 3rdparty/nanoflann/) # nanoflann licence
install(DIRECTORY ${eos_3RDPARTY_DIR}/eigen3-nnls/src/ DESTINATION 3rdparty/eigen3-nnls/src) # eigen3-nnls header
install(FILES ${eos_3RDPARTY_DIR}/eigen3-nnls/README.md DESTINATION 3rdparty/eigen3-nnls/) # eigen3-nnls attribution
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
