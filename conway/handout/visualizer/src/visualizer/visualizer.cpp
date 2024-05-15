#include <iostream>
#include <ctime>

#include "scene.hpp"
#include "world.hpp"

int main(int argc, char *argv[])
{
	hpcgame::World3D world1(512, 0), world2(512, 0);
	world1.generate(0.5);
	hpcgame::visualizer::GLScene(world1, world2, 1300, 1300, argc, argv);
	glutMainLoop();
}
