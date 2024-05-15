add_library(gameworld world.cpp)
target_include_directories(gameworld PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gameworld PRIVATE OpenMP::OpenMP_CXX)