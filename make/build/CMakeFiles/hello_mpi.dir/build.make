# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lxy/hpcgame2024/make

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lxy/hpcgame2024/make/build

# Include any dependencies generated for this target.
include CMakeFiles/hello_mpi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hello_mpi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello_mpi.dir/flags.make

CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o: CMakeFiles/hello_mpi.dir/flags.make
CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o: ../hello_mpi.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lxy/hpcgame2024/make/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o -c /home/lxy/hpcgame2024/make/hello_mpi.cpp

CMakeFiles/hello_mpi.dir/hello_mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hello_mpi.dir/hello_mpi.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lxy/hpcgame2024/make/hello_mpi.cpp > CMakeFiles/hello_mpi.dir/hello_mpi.cpp.i

CMakeFiles/hello_mpi.dir/hello_mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hello_mpi.dir/hello_mpi.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lxy/hpcgame2024/make/hello_mpi.cpp -o CMakeFiles/hello_mpi.dir/hello_mpi.cpp.s

# Object files for target hello_mpi
hello_mpi_OBJECTS = \
"CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o"

# External object files for target hello_mpi
hello_mpi_EXTERNAL_OBJECTS =

hello_mpi: CMakeFiles/hello_mpi.dir/hello_mpi.cpp.o
hello_mpi: CMakeFiles/hello_mpi.dir/build.make
hello_mpi: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
hello_mpi: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
hello_mpi: CMakeFiles/hello_mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lxy/hpcgame2024/make/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello_mpi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello_mpi.dir/build: hello_mpi

.PHONY : CMakeFiles/hello_mpi.dir/build

CMakeFiles/hello_mpi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello_mpi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello_mpi.dir/clean

CMakeFiles/hello_mpi.dir/depend:
	cd /home/lxy/hpcgame2024/make/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lxy/hpcgame2024/make /home/lxy/hpcgame2024/make /home/lxy/hpcgame2024/make/build /home/lxy/hpcgame2024/make/build /home/lxy/hpcgame2024/make/build/CMakeFiles/hello_mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hello_mpi.dir/depend

