cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR)
project(GameOfLife LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -O3")

find_package(OpenMP REQUIRED)

if (USE_VISUALIZATION)
    set(OpenGL_GL_PREFERENCE "LEGACY")

    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)
endif()

# collect all bin and lib directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(3RDPARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

include_directories(${3RDPARTY_DIR}/argparse/include)

add_subdirectory(src)