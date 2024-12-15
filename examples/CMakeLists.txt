# The examples need a few additional dependencies (e.g. boost filesystem, program_options, and OpenCV highgui):
find_package(OpenCV 4 CONFIG REQUIRED core imgproc imgcodecs)
find_package(cxxopts CONFIG REQUIRED)

# Simple model fitting (orthographic camera & shape to landmarks) example:
add_executable(fit-model-simple fit-model-simple.cpp)
target_link_libraries(fit-model-simple PRIVATE cxxopts::cxxopts eos opencv_core opencv_imgproc opencv_imgcodecs)
target_link_libraries(fit-model-simple PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_compile_options(fit-model-simple PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours:
add_executable(fit-model fit-model.cpp)
target_link_libraries(fit-model PRIVATE cxxopts::cxxopts eos opencv_core opencv_imgproc opencv_imgcodecs)
target_link_libraries(fit-model PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_compile_options(fit-model PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")

# Model fitting example that fits orthographic camera, shape, blendshapes, and contours to multiple images:
add_executable(fit-model-multi fit-model-multi.cpp)
target_link_libraries(fit-model-multi PRIVATE cxxopts::cxxopts eos opencv_core opencv_imgproc opencv_imgcodecs)
target_link_libraries(fit-model-multi PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-pthread>$<$<CXX_COMPILER_ID:Clang>:-pthreads>")
target_compile_options(fit-model-multi PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")

# Generate random samples from the model:
add_executable(generate-obj generate-obj.cpp)
target_link_libraries(generate-obj PRIVATE cxxopts::cxxopts eos opencv_core opencv_imgcodecs)

# Install these targets:
install(TARGETS fit-model-simple DESTINATION bin)
install(TARGETS fit-model DESTINATION bin)
install(TARGETS fit-model-multi DESTINATION bin)
install(TARGETS generate-obj DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION bin)


if(EOS_BUILD_CERES_EXAMPLE)
  # Find Ceres, for the fit-model-ceres app:
  find_package(Ceres CONFIG REQUIRED)

  # Single and multi-image non-linear model fitting with Ceres example:
  add_executable(fit-model-ceres fit-model-ceres.cpp)
  target_link_libraries(fit-model-ceres PRIVATE cxxopts::cxxopts eos Ceres::ceres opencv_core opencv_imgproc opencv_imgcodecs)
  target_compile_options(fit-model-ceres PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
  install(TARGETS fit-model-ceres DESTINATION bin)
endif()
