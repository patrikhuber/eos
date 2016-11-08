project(utils)
cmake_minimum_required(VERSION 2.8.12)

# The utils need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):

#check installed version in order to include the correct OpenCV libraries
#version variable is defined from project root's CMakeLists
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
  message(STATUS "OpenCV 2.x detected")
  find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
  message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
  find_package(OpenCV 3.0.0 REQUIRED core imgproc imgcodecs)
endif()
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if(MSVC)
	# The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
	add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
	add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal ${OpenCV_LIBS} ${Boost_LIBRARIES})


# Store a json file as cereal .bin:
add_executable(json-to-cereal-binary json-to-cereal-binary.cpp)
target_link_libraries(json-to-cereal-binary ${OpenCV_LIBS} ${Boost_LIBRARIES})
# Generate python bindings using pybind11:
if(GENERATE_PYTHON_BINDINGS)
	pybind11_add_module(python-bindings generate-python-bindings.cpp pybind11_glm.hpp)
	target_link_libraries(python-bindings PRIVATE ${OpenCV_LIBS} ${Boost_LIBRARIES})
	set_target_properties(python-bindings PROPERTIES OUTPUT_NAME eos)
	install(TARGETS python-bindings DESTINATION bin)
endif()

# install target:
install(TARGETS scm-to-cereal DESTINATION bin)
install(TARGETS json-to-cereal-binary DESTINATION bin)
