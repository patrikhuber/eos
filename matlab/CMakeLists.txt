cmake_minimum_required(VERSION 3.7.0)

# If Matlab_ROOT_DIR is set, the Matlab at that location is used.
find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)

# Dependencies of the modules:
#check installed version in order to include the correct OpenCV libraries
#version variable is defined from project root's CMakeLists
if("${OpenCV_VERSION_MAJOR}$" EQUAL 2)
  message(STATUS "OpenCV 2.x detected")
  find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui)
elseif("${OpenCV_VERSION_MAJOR}$" EQUAL 3)
  message(STATUS "OpenCV 3.x detected - including imgcodecs for compatibility")
  find_package(OpenCV 3 REQUIRED core imgproc imgcodecs)
endif()
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if(MSVC)
  # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
  add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
  add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system filesystem REQUIRED) # Why do we need boost for MorphableModel.hpp?

# Our helper-headers, and Matlab-specific headers - to make them show up in IDEs:
set(EOS_MATLAB_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mexplus_eigen.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mexplus_eos_types.hpp
)
add_custom_target(eos-matlab-headers SOURCES ${EOS_MATLAB_HEADERS})

# See: https://cmake.org/cmake/help/v3.7/module/FindMatlab.html?highlight=findmatlab#command:matlab_add_mex
matlab_add_mex(
  NAME eos-matlab-fitting
  #[EXECUTABLE | MODULE | SHARED] # SHARED is the default.
  SRC +eos/+fitting/private/fitting.cpp
  OUTPUT_NAME fitting #[OUTPUT_NAME output_name]
  # DOCUMENTATION +eos/+fitting/fit_shape_and_pose.m # doesn't work - wrong path probably. But it renames the file to fitting.m, so not what we want anyway.
  LINK_TO eos ${OpenCV_LIBS} ${Boost_LIBRARIES} #[LINK_TO target1 target2 ...]
  #[...]
)

target_include_directories(eos-matlab-fitting PRIVATE ${eos_3RDPARTY_DIR}/mexplus/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/demo.m DESTINATION matlab)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION matlab)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/+eos DESTINATION matlab PATTERN "*.cpp" EXCLUDE)
install(TARGETS eos-matlab-fitting DESTINATION matlab/+eos/+fitting/private)
