add_subdirectory(world)
add_subdirectory(generator)
add_subdirectory(runner)

if(USE_VISUALIZATION)
    add_subdirectory(visualizer)
endif()