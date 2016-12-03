project(matlab-bindings)
cmake_minimum_required(VERSION 3.7.0)

# If Matlab_ROOT_DIR is set, the Matlab at that location is used.
find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)

# See: https://cmake.org/cmake/help/v3.7/module/FindMatlab.html?highlight=findmatlab#command:matlab_add_mex
matlab_add_mex(
    NAME eos_fitting
    #[EXECUTABLE | MODULE | SHARED] # SHARED is the default.
    SRC +eos/+fitting/private/fitting.cpp
    OUTPUT_NAME fitting #[OUTPUT_NAME output_name]
    # DOCUMENTATION +eos/+fitting/fit_shape_and_pose.m # doesn't work - wrong path probably. But it renames the file to fitting.m, so not what we want anyway.
    #[LINK_TO target1 target2 ...] # OpenCV etc?
    #[...]
)

target_include_directories(eos_fitting PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/mexplus/include ${CMAKE_SOURCE_DIR}/matlab/include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/matlab/include DESTINATION matlab)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/matlab/+eos DESTINATION matlab PATTERN "*.cpp" EXCLUDE)
install(TARGETS eos_fitting DESTINATION matlab/+eos/+fitting/private)

# Todo: Look at opencv mex... eg Rect, the .c file. Do they do all the dispatching in mexFunc()?
# Also this may have wrapping for std::vector<T>: https://github.com/kyamagu/mexopencv/blob/master/include/MxArray.hpp