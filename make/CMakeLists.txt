cmake_minimum_required(VERSION 3.10)
project(HelloWorlds LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(hello_omp hello_omp.cpp)

if(OpenMP_CXX_FOUND)
	target_link_libraries(hello_omp PUBLIC OpenMP::OpenMP_CXX)
endif()


add_executable(hello_mpi hello_mpi.cpp)
if(MPI_CXX_FOUND)
	include_directories(${MPI_INCLUDE_PATH})
	target_link_libraries(hello_mpi ${MPI_LIBRARIES})
endif()
add_executable(hello_cuda hello_cuda.cu)

