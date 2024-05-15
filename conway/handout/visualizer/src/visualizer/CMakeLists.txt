add_executable(visualizer visualizer.cpp scene.cpp scene.hpp)
target_include_directories(visualizer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(visualizer PUBLIC OpenGL::OpenGL GLUT::GLUT GLU)
target_link_libraries(visualizer PUBLIC gameworld generator generator_cuda )