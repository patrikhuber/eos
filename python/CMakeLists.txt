project(python-bindings)
cmake_minimum_required(VERSION 2.8.12)

# The python bindings may need a few additional dependencies (e.g. boost filesystem and OpenCV), which are
# included below - at some point it should be checked if they are really needed.

# Check installed version in order to include the correct OpenCV libraries
# OpenCV_VERSION_MAJOR is defined from the project root's CMakeLists
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
  message(STATUS "OpenCV 2.x detected")
  find_package(OpenCV 2.4.3 REQUIRED core imgproc)
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

pybind11_add_module(python-bindings generate-python-bindings.cpp pybind11_glm.hpp)
target_link_libraries(python-bindings PRIVATE eos ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(python-bindings PROPERTIES OUTPUT_NAME eos)
install(TARGETS python-bindings DESTINATION python)
