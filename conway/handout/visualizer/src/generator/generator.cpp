#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

#include "world.hpp"
namespace hpcgame {

void update_world_cuda(World3D& old_world, World3D& new_world, const size_t step);

size_t count_neighbor(const World3D& world, int x, int y, int z) {
    auto curr_space = world.get_data();
    const size_t M = world.get_size();

    const size_t lx = ((x + M - 1) % M) * M * M;
    const size_t mx = x * M * M;
    const size_t rx = ((x + 1) % M) * M * M;

    const size_t ly = ((y + M - 1) % M) * M;
    const size_t my = y * M;
    const size_t ry = ((y + 1) % M) * M;

    const size_t lz = (z + M - 1) % M;
    const size_t mz = z;
    const size_t rz = (z + 1) % M;

    return curr_space[lx + ly + lz] + curr_space[lx + ly + mz] +
           curr_space[lx + ly + rz] + curr_space[lx + my + lz] +
           curr_space[lx + my + mz] + curr_space[lx + my + rz] +
           curr_space[lx + ry + lz] + curr_space[lx + ry + mz] +
           curr_space[lx + ry + rz] + curr_space[mx + ly + lz] +
           curr_space[mx + ly + mz] + curr_space[mx + ly + rz] +
           curr_space[mx + my + lz] + curr_space[mx + my + rz] +
           curr_space[mx + ry + lz] + curr_space[mx + ry + mz] +
           curr_space[mx + ry + rz] + curr_space[rx + ly + lz] +
           curr_space[rx + ly + mz] + curr_space[rx + ly + rz] +
           curr_space[rx + my + lz] + curr_space[rx + my + mz] +
           curr_space[rx + my + rz] + curr_space[rx + ry + lz] +
           curr_space[rx + ry + mz] + curr_space[rx + ry + rz];
}

void update_state(const World3D& old_world, World3D& new_world, int x, int y, int z) {
    auto curr_space = old_world.get_data();
    auto next_space = new_world.get_data();
    const size_t M = old_world.get_size();

    const int neighbor_count = count_neighbor(old_world, x, y, z);
    const uint8_t curr_state = curr_space[x * M * M + y * M + z];
    uint8_t &next_state = next_space[x * M * M + y * M + z];

    if (curr_state == 1) {
        if (neighbor_count < 5 || neighbor_count > 7)
            next_state = 0;
        else
            next_state = 1;
    } else {
        if (neighbor_count == 6) {
            next_state = 1;
        } else {
            next_state = 0;
        }
    }
}

void update_world(World3D& old_world, World3D& new_world, const size_t step) {
    const size_t M = old_world.get_size();
    for (size_t t = 0; t < step; t++) {
        for (size_t x = 0; x < M; x++) {
            for (size_t y = 0; y < M; y++) {
                for (size_t z = 0; z < M; z++) {
                    update_state(old_world, new_world, x, y, z);
                }
            }
        }
        std::swap(old_world, new_world);
    }
}

}