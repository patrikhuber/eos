cmake_minimum_required(VERSION 3.7.0)

# If Matlab_ROOT_DIR is set, the Matlab at that location is used.
find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)

# Our helper-headers, and Matlab-specific headers - to make them show up in IDEs:
set(EOS_MATLAB_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mexplus_eigen.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/mexplus_eos_types.hpp
)
add_custom_target(eos-matlab-headers SOURCES ${EOS_MATLAB_HEADERS})

# See: https://cmake.org/cmake/help/latest/module/FindMatlab.html?#command:matlab_add_mex
matlab_add_mex(
  NAME eos-matlab-fitting
  #[EXECUTABLE | MODULE | SHARED] # SHARED is the default.
  SRC +eos/+fitting/private/fitting.cpp
  OUTPUT_NAME fitting
  # DOCUMENTATION +eos/+fitting/fit_shape_and_pose.m # doesn't work - wrong path probably. But it renames the file to fitting.m, so not what we want anyway.
  LINK_TO eos
  #[...]
)

#matlab_add_mex(
#  NAME eos-matlab-render
#  SRC +eos/+render/private/render.cpp
#  OUTPUT_NAME render
#  LINK_TO eos
#)

# Group the matlab bindings targets into one folder (in IDEs):
set_target_properties(eos-matlab-headers eos-matlab-fitting PROPERTIES FOLDER "matlab-bindings") # eos-matlab-render

target_include_directories(eos-matlab-fitting PRIVATE ${eos_3RDPARTY_DIR}/mexplus/include ${CMAKE_CURRENT_SOURCE_DIR}/include) # the latter one we can use the eos-matlab-headers target?
# target_include_directories(eos-matlab-render PRIVATE ${eos_3RDPARTY_DIR}/mexplus/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/demo.m DESTINATION matlab)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION matlab)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/+eos DESTINATION matlab PATTERN "*.cpp" EXCLUDE)
install(TARGETS eos-matlab-fitting DESTINATION matlab/+eos/+fitting/private)
# install(TARGETS eos-matlab-render DESTINATION matlab/+eos/+render/private)
