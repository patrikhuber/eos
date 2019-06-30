# scm-to-cereal needs boost::program_options:
if(MSVC)
  # The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
  add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
  add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
  add_definitions(-D_HAS_AUTO_PTR_ETC) # Boost 1.65.1 still does not work with VS C++17 mode, this is the workaround
endif()
find_package(Boost 1.50.0 COMPONENTS system program_options REQUIRED)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal eos ${Boost_LIBRARIES})
target_include_directories(scm-to-cereal PUBLIC ${Boost_INCLUDE_DIRS})

# Install targets:
install(TARGETS scm-to-cereal DESTINATION bin)
