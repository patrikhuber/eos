find_package(Boost 1.71.0 REQUIRED COMPONENTS program_options)
message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal PRIVATE eos Boost::program_options)

# Install targets:
install(TARGETS scm-to-cereal DESTINATION bin)
