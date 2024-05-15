add_executable(runner runner.cpp)
target_include_directories(runner PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(runner PUBLIC gameworld generator generator_cuda)
target_include_directories(runner PUBLIC ${3RDPARTY_DIR}/argparse/include)