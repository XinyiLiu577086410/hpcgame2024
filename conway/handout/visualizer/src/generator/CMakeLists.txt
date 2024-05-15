add_library(generator_cuda STATIC generator.cu)
target_link_libraries(generator_cuda PUBLIC gameworld)

add_library(generator STATIC generator.cpp)
target_link_libraries(generator PUBLIC gameworld generator_cuda)