project(eos)
cmake_minimum_required(VERSION 2.8.10)
set(eos_VERSION_MAJOR 0)
set(eos_VERSION_MINOR 6)
set(eos_VERSION_PATCH 1)
set(eos_VERSION ${eos_VERSION_MAJOR}.${eos_VERSION_MINOR}.${eos_VERSION_PATCH})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Check if a supported compiler is used and add c++11/14 flag
include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
		message(FATAL_ERROR "Need at least gcc 4.8 to compile.")
	elseif(CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 4.8 OR (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9))
		# This compiles with a warning at the moment, but support for gcc 4.8.x will be removed in the very near future.
		check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
		if(HAS_CXX11_FLAG)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
		endif()
		message(WARNING "Support for gcc 4.8 will be removed in the very near future. Please upgrade your compiler.")
	else() # gcc version is >4.8
		check_cxx_compiler_flag(-std=c++14 HAS_CXX14_FLAG)
		if(HAS_CXX14_FLAG)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
		endif()
	endif()
	# Note: gcc is fine without -pthreads.
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC") # the quotes are needed here, maybe because "MSVC" seems to be a keyword
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
		message(FATAL_ERROR "Visual Studio 2015 or newer is required.")
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	# >=3.5 works, not sure about older versions. Also, libstdc++ from gcc >=4.9 may be needed.
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
		message(WARNING "Clang below version 3.5 may or may not work. Please upgrade your compiler.")
	endif()
	check_cxx_compiler_flag(-std=c++14 HAS_CXX14_FLAG)
	if(HAS_CXX14_FLAG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	endif()
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthreads")
	# Eigen::LevenbergMarquardt probably needs -pthreads.
	# Cleaner way would be to add it to fit-model's target_link_libraries, but that requires a CMake >2.8.10.
else() # no GNU, no MSVC, no Clang
	message(WARNING "You are using an unsupported compiler. Compilation has only been tested with MSVC, GCC and Clang.")
	check_cxx_compiler_flag(-std=c++14 HAS_CXX14_FLAG)
	if(HAS_CXX14_FLAG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	endif()
endif()

# All the options for building the library. Can be changed on the command-line or in initial_cache.cmake.
message(STATUS "Options:")
option(BUILD_EXAMPLES "Build the example applications." ON)
message(STATUS "BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
option(BUILD_DOCUMENTATION "Build the library documentation." OFF)
message(STATUS "BUILD_DOCUMENTATION: ${BUILD_DOCUMENTATION}")

# Build a CPack driven installer package:
include(InstallRequiredSystemLibraries) # This module will include any runtime libraries that are needed by the project for the current platform
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${eos_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${eos_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${eos_VERSION_PATCH}")
include(CPack)

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.50.0 COMPONENTS system REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/cereal-1.1.1/include")
set(glm_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/glm-0.9.7.0")

# Header files:
set(HEADERS
	include/eos/core/Landmark.hpp
	include/eos/core/LandmarkMapper.hpp
	include/eos/morphablemodel/PcaModel.hpp
	include/eos/morphablemodel/MorphableModel.hpp
	include/eos/morphablemodel/Blendshape.hpp
	include/eos/morphablemodel/io/cvssp.hpp
	include/eos/morphablemodel/io/mat_cerealisation.hpp
	include/eos/fitting/affine_camera_estimation.hpp
	include/eos/fitting/nonlinear_camera_estimation.hpp
	include/eos/fitting/detail/nonlinear_camera_estimation_detail.hpp
	include/eos/fitting/linear_shape_fitting.hpp
	include/eos/fitting/contour_correspondence.hpp
	include/eos/fitting/blendshape_fitting.hpp
	include/eos/fitting/fitting.hpp
	include/eos/render/Mesh.hpp
	include/eos/render/utils.hpp
	include/eos/render/render.hpp
	include/eos/render/render_affine.hpp
	include/eos/render/detail/render_detail.hpp
	include/eos/render/detail/render_affine_detail.hpp
	include/eos/render/texture_extraction.hpp
	include/eos/render/detail/texture_extraction_detail.hpp
)

# Add header includes:
include_directories("include")
include_directories(${CEREAL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${glm_INCLUDE_DIR})

# Custom target for the library, to make the headers show up in IDEs:
add_custom_target(eos SOURCES ${HEADERS})
source_group(core include/eos/core/*)
source_group(morphablemodel include/eos/morphablemodel/*)
source_group(morphablemodel\\io include/eos/morphablemodel/io/*)
source_group(fitting include/eos/fitting/*)
source_group(fitting\\detail include/eos/fitting/detail/*)
source_group(render include/eos/render/*)
source_group(render\\detail include/eos/render/detail/*)

# The install target:
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include) # our library headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/share/ DESTINATION share) # the model and metadata
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/cereal-1.1.1/ DESTINATION 3rdparty/cereal-1.1.1) # cereal headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/glm-0.9.7.0/ DESTINATION 3rdparty/glm-0.9.7.0) # glm headers

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(BUILD_DOCUMENTATION)
	add_subdirectory(doc)
endif()
