# CMake config file for the eos library
# -------------------------------------
# It defines the following variables:
#  eos_INCLUDE_DIRS - the eos include directory
# and the target 'eos' to link against.

# Get the absolute path of this file:
get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(eos_INCLUDE_DIRS "${SELF_DIR}/include")

# Defines our library target and its dependencies:
include(${SELF_DIR}/eos-targets.cmake)
