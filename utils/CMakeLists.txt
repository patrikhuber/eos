find_package(cxxopts CONFIG REQUIRED)

# Converts a CVSSP .scm Morphable Model to a cereal binary file:
add_executable(scm-to-cereal scm-to-cereal.cpp)
target_link_libraries(scm-to-cereal PRIVATE cxxopts::cxxopts eos)

# Install targets:
install(TARGETS scm-to-cereal DESTINATION bin)
