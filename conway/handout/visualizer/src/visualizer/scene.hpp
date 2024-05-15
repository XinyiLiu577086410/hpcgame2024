#pragma once

#include "world.hpp"

#include <ctime>

#include <GL/glut.h>

namespace hpcgame::visualizer {

void GLScene(World3D world1, World3D world2, int, int, int argc, char*argv[]);
void Cleanup();

void DisplayGL();
void KeyboardGL(unsigned char c, int x, int y);
void ReshapeGL(int w, int h);

void render();

}