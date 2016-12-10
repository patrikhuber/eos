project(examples)
cmake_minimum_required(VERSION 2.8.10)

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):

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

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthreads") # IF GCC OR CLANG!
# Eigen::LevenbergMarquardt probably needs -pthreads. => So eos needs pthreads! How to add in modern CMake?
# Cleaner way would be to add it to fit-model's target_link_libraries, but that requires a CMake >2.8.10.
# clang and gcc.
# Or if we include Eigen in a modern way, it may include it as a transitive dependency and the problem may go away.

# Simple model fitting (orthographic camera & shape to landmarks) example:
add_executable(fit-model-simple fit-model-simple.cpp)
target_link_libraries(fit-model-simple eos ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours:
add_executable(fit-model fit-model.cpp)
target_link_libraries(fit-model ${OpenCV_LIBS} ${Boost_LIBRARIES})

if(EOS_BUILD_CERES_EXAMPLE)
	# Find Ceres, for the fit-model-ceres app:
	find_package(Ceres REQUIRED)
	message(STATUS "Ceres locations: Headers: ${CERES_INCLUDE_DIRS} Library: ${CERES_LIBRARIES}")
	include_directories(${CERES_INCLUDE_DIRS})

	# Single and multi-image non-linear model fitting with Ceres example:
	add_executable(fit-model-ceres fit-model-ceres.cpp)
	target_link_libraries(fit-model-ceres ${CERES_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
	install(TARGETS fit-model-ceres DESTINATION bin)
endif()

# Generate random samples from the model:
add_executable(generate-obj generate-obj.cpp)
target_link_libraries(generate-obj ${OpenCV_LIBS} ${Boost_LIBRARIES})


# install target:
install(TARGETS fit-model-simple DESTINATION bin)
install(TARGETS fit-model DESTINATION bin)
install(TARGETS generate-obj DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION bin)
