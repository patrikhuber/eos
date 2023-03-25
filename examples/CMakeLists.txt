# The examples need a few additional dependencies (e.g. boost filesystem, program_options, and OpenCV highgui):

# Check installed version in order to include the correct OpenCV libraries:
# First call find_package without a version to find any OpenCV. OpenCV_VERSION_MAJOR is then defined.
find_package(OpenCV REQUIRED core)
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
  message(STATUS "OpenCV 2.x detected")
  find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
  message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
  find_package(OpenCV 3 REQUIRED core imgproc imgcodecs)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 4)
  message(STATUS "OpenCV 4.x detected - including imgcodecs for compatibility")
  find_package(OpenCV 4 REQUIRED core imgproc imgcodecs)
endif()
# This allows us to compile in RelWithDebInfo. It'll use the Release-version of OpenCV:
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

set(Boost_NO_WARN_NEW_VERSIONS ON) # Supress "New Boost version may have incorrect dependencies or import targets" warning
find_package(Boost 1.71.0 REQUIRED COMPONENTS filesystem program_options)

# Simple model fitting (orthographic camera & shape to landmarks) example:
add_executable(fit-model-simple fit-model-simple.cpp)
target_link_libraries(fit-model-simple PRIVATE eos ${OpenCV_LIBS} Boost::filesystem Boost::program_options)
target_link_libraries(fit-model-simple PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model-simple PRIVATE ${OpenCV_INCLUDE_DIRS})

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours:
add_executable(fit-model fit-model.cpp)
target_link_libraries(fit-model PRIVATE eos ${OpenCV_LIBS} Boost::filesystem Boost::program_options)
target_link_libraries(fit-model PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model PRIVATE ${OpenCV_INCLUDE_DIRS})

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours to multiple images:
add_executable(fit-model-multi fit-model-multi.cpp)
target_link_libraries(fit-model-multi PRIVATE eos ${OpenCV_LIBS} Boost::filesystem Boost::program_options)
target_link_libraries(fit-model-multi PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_include_directories(fit-model-multi PRIVATE ${OpenCV_INCLUDE_DIRS})

# Generate random samples from the model:
add_executable(generate-obj generate-obj.cpp)
target_link_libraries(generate-obj PRIVATE eos ${OpenCV_LIBS} Boost::filesystem Boost::program_options)
target_include_directories(generate-obj PRIVATE ${OpenCV_INCLUDE_DIRS})

# Install these targets:
install(TARGETS fit-model-simple DESTINATION bin)
install(TARGETS fit-model DESTINATION bin)
install(TARGETS fit-model-multi DESTINATION bin)
install(TARGETS generate-obj DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION bin)


if(EOS_BUILD_CERES_EXAMPLE)
  # Find Ceres, for the fit-model-ceres app:
  find_package(Ceres REQUIRED)

  # Single and multi-image non-linear model fitting with Ceres example:
  add_executable(fit-model-ceres fit-model-ceres.cpp)
  target_link_libraries(fit-model-ceres PRIVATE eos Ceres::ceres ${OpenCV_LIBS} Boost::filesystem Boost::program_options)
  target_include_directories(fit-model-ceres PRIVATE ${OpenCV_INCLUDE_DIRS})
  install(TARGETS fit-model-ceres DESTINATION bin)
endif()
