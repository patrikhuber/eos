find_package(Doxygen REQUIRED)

set(doxyfile_in ${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in)
set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${doxyfile_in} ${doxyfile} @ONLY)

add_custom_target(doc
  COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen"
  VERBATIM
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION doc)
