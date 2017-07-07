# The python bindings may need a few additional dependencies (e.g. boost filesystem), which are
# included below - at some point it should be checked if they are really needed.

if(MSVC)
  # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
  add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
  add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.50.0 COMPONENTS system filesystem program_options REQUIRED)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

pybind11_add_module(python-bindings generate-python-bindings.cpp pybind11_glm.hpp pybind11_Image.hpp)
target_link_libraries(python-bindings PRIVATE eos ${Boost_LIBRARIES})
set_target_properties(python-bindings PROPERTIES OUTPUT_NAME eos)

install(TARGETS python-bindings DESTINATION python)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/demo.py DESTINATION python)
