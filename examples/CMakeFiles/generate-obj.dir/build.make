# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ml/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/ml/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ml/eos

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ml/eos

# Include any dependencies generated for this target.
include examples/CMakeFiles/generate-obj.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/generate-obj.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/generate-obj.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/generate-obj.dir/flags.make

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o: examples/CMakeFiles/generate-obj.dir/flags.make
examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o: examples/generate-obj.cpp
examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o: examples/CMakeFiles/generate-obj.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/ml/eos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o"
	cd /home/ml/eos/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o -MF CMakeFiles/generate-obj.dir/generate-obj.cpp.o.d -o CMakeFiles/generate-obj.dir/generate-obj.cpp.o -c /home/ml/eos/examples/generate-obj.cpp

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/generate-obj.dir/generate-obj.cpp.i"
	cd /home/ml/eos/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ml/eos/examples/generate-obj.cpp > CMakeFiles/generate-obj.dir/generate-obj.cpp.i

examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/generate-obj.dir/generate-obj.cpp.s"
	cd /home/ml/eos/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ml/eos/examples/generate-obj.cpp -o CMakeFiles/generate-obj.dir/generate-obj.cpp.s

# Object files for target generate-obj
generate__obj_OBJECTS = \
"CMakeFiles/generate-obj.dir/generate-obj.cpp.o"

# External object files for target generate-obj
generate__obj_EXTERNAL_OBJECTS =

examples/generate-obj: examples/CMakeFiles/generate-obj.dir/generate-obj.cpp.o
examples/generate-obj: examples/CMakeFiles/generate-obj.dir/build.make
examples/generate-obj: /usr/local/lib/libopencv_imgcodecs.so.4.7.0
examples/generate-obj: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
examples/generate-obj: /usr/lib/x86_64-linux-gnu/libboost_program_options.so.1.71.0
examples/generate-obj: /usr/local/lib/libopencv_imgproc.so.4.7.0
examples/generate-obj: /usr/local/lib/libopencv_core.so.4.7.0
examples/generate-obj: examples/CMakeFiles/generate-obj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/ml/eos/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable generate-obj"
	cd /home/ml/eos/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generate-obj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/generate-obj.dir/build: examples/generate-obj
.PHONY : examples/CMakeFiles/generate-obj.dir/build

examples/CMakeFiles/generate-obj.dir/clean:
	cd /home/ml/eos/examples && $(CMAKE_COMMAND) -P CMakeFiles/generate-obj.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/generate-obj.dir/clean

examples/CMakeFiles/generate-obj.dir/depend:
	cd /home/ml/eos && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ml/eos /home/ml/eos/examples /home/ml/eos /home/ml/eos/examples /home/ml/eos/examples/CMakeFiles/generate-obj.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : examples/CMakeFiles/generate-obj.dir/depend

