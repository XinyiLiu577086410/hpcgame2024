#pragma once

#include "world.hpp"

namespace hpcgame {

void update_world_cuda(World3D& old_world, World3D& new_world, const size_t step = 1);

size_t count_neighbor(const World3D& world, int x, int y, int z);

void update_state(const World3D& old_world, World3D& new_world, int x, int y, int z);

void update_world(World3D& old_world, World3D& new_world, const size_t step = 1);

}