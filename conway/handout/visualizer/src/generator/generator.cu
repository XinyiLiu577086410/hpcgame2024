#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

#include <cuda_runtime.h>

#include "world.hpp"

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define BYTES_PER_THREAD 1

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

inline void syncAndCheck(const char* const file, int const line, bool force_check = false) {
#ifdef DEBUG
    force_check = true;
#endif
    if (force_check) {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " "
                                    + file + ":" + std::to_string(line) + " \n");
        }
    }
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__, false)
#define sync_check_cuda_error_force() syncAndCheck(__FILE__, __LINE__, true)

namespace hpcgame {

void update_world_cuda(World3D& old_world, World3D& new_world, const size_t step) {
  throw std::runtime_error("CUDA not implemented");
}

}