project(eos)
cmake_minimum_required(VERSION 2.8.7)
set(eos_VERSION_MAJOR 0)
set(eos_VERSION_MINOR 1)
set(eos_VERSION_PATCH 0)
set(eos_VERSION ${eos_VERSION_MAJOR}.${eos_VERSION_MINOR}.${eos_VERSION_PATCH})

# g++ needs a compiler flag to enable C++11 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 HAS_CXX11_FLAG)
if (HAS_CXX11_FLAG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

# All the options for building the library. Can be changed on the command-line or in initial_cache.cmake.
message(STATUS "Options:")
option(BUILD_EXAMPLES "Build the example applications." ON)
message(STATUS "BUILD_EXAMPLES: ${BUILD_EXAMPLES}")
option(BUILD_DOCUMENTATION "Build the library documentation." ON)
message(STATUS "BUILD_DOCUMENTATION: ${BUILD_DOCUMENTATION}")

# Build a CPack driven installer package:
include(InstallRequiredSystemLibraries) # This module will include any runtime libraries that are needed by the project for the current platform
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${eos_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${eos_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${eos_VERSION_PATCH}")
include(CPack)

# Set debug library postfix:
set(CMAKE_DEBUG_POSTFIX "-d")

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.48.0 COMPONENTS system REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

# Source and header files:
set(SOURCES
	src/eos/core/LandmarkMapper.cpp
	src/eos/morphablemodel/PcaModel.cpp
	src/eos/morphablemodel/MorphableModel.cpp
	src/eos/morphablemodel/io/cvssp.cpp
	src/eos/fitting/AffineCameraEstimation.cpp
	src/eos/fitting/LinearShapeFitting.cpp
	src/eos/render/Mesh.cpp
	src/eos/render/utils.cpp
)

set(HEADERS
	include/eos/core/LandmarkMapper.hpp
	include/eos/morphablemodel/PcaModel.hpp
	include/eos/morphablemodel/MorphableModel.hpp
	include/eos/morphablemodel/io/cvssp.hpp
	include/eos/fitting/AffineCameraEstimation.hpp
	include/eos/fitting/LinearShapeFitting.hpp
	include/eos/render/Mesh.hpp
	include/eos/render/utils.hpp
)

# Add header includes:
include_directories("include")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Make the library:
add_library(eos ${SOURCES} ${HEADERS})
target_link_libraries(eos ${OpenCV_LIBS} ${Boost_LIBRARIES})

# The install target:
install(TARGETS eos DESTINATION ${CMAKE_INSTALL_PREFIX}/lib EXPORT eos-targets)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/share/ DESTINATION ${CMAKE_INSTALL_PREFIX}/share)

# Export the target:
install(EXPORT eos-targets DESTINATION ${CMAKE_INSTALL_PREFIX}/) # creates eos-targets.cmake
install(FILES eos-config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/)
configure_file(eos-config-version.cmake.in "${PROJECT_BINARY_DIR}/eos-config-version.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/eos-config-version.cmake" DESTINATION ${CMAKE_INSTALL_PREFIX}/)

if(BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

if(BUILD_DOCUMENTATION)
	add_subdirectory(doc)
endif()
