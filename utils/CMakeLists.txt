cmake_minimum_required(VERSION 3.1.3)
project(utils)

# The utils need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):

#check installed version in order to include the correct OpenCV libraries
#version variable is defined from project root's CMakeLists
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
  message(STATUS "OpenCV 2.x detected")
  find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
  message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
  find_package(OpenCV 3 REQUIRED core imgproc imgcodecs)
endif()
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if(MSVC)
  # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
  add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
  add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system filesystem program_options REQUIRED)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal eos ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Converts a file created with share/convert_bfm2009_to_raw_binary.m to a cereal binary file:
add_executable(bfm-binary-to-cereal bfm-binary-to-cereal.cpp)
target_link_libraries(bfm-binary-to-cereal eos ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Reads an edgestruct CSV file created from Matlab, and converts it to json:
add_executable(edgestruct-csv-to-json edgestruct-csv-to-json.cpp)
target_link_libraries(edgestruct-csv-to-json eos ${Boost_LIBRARIES})

# Store a json file as cereal .bin:
add_executable(json-to-cereal-binary json-to-cereal-binary.cpp)
target_link_libraries(json-to-cereal-binary eos ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install target:
install(TARGETS scm-to-cereal DESTINATION bin)
install(TARGETS bfm-binary-to-cereal DESTINATION bin)
install(TARGETS json-to-cereal-binary DESTINATION bin)
install(TARGETS edgestruct-csv-to-json DESTINATION bin)
