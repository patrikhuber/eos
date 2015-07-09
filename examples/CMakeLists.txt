project(examples)
cmake_minimum_required(VERSION 2.8.7)

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

#include_directories(${eos_SOURCE_DIR})

# Model fitting (affine cam & shape to landmarks) example:
add_executable(fit_model fit_model.cpp)
target_link_libraries(fit_model ${OpenCV_LIBS} ${Boost_LIBRARIES})
# TODO Add dependency to library-target??

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install target:
install(TARGETS fit_model DESTINATION bin)
install(TARGETS scm-to-cereal DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data DESTINATION bin)
