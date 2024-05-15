#include <iostream>
#include <ctime>
#include <argparse/argparse.hpp>

#include "world.hpp"
#include "generator.hpp"

int main(int argc, char *argv[])
{
	// Parse arguments
	argparse::ArgumentParser program("Game of Life");
	program.add_argument("-s", "--size")
		.help("Size of the world")
		.default_value(512)
		.action([](const std::string &value) { return std::stoi(value); });
	program.add_argument("-n", "--step")
		.help("Number of steps")
		.default_value(100)
		.action([](const std::string &value) { return std::stoi(value); });
	program.add_argument("-o", "--output")
		.help("Output file")
		.default_value("output.bin");
	program.add_argument("-i", "--input")
		.help("Input file")
		.default_value("");

	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error &err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		exit(1);
	}

	auto size = program.get<int>("--size");
	auto iterations = program.get<int>("--step");
	auto output = program.get<std::string>("--output");
	auto input = program.get<std::string>("--input");

	// Create world
	hpcgame::World3D world1, world2;

	if (input.empty()) {
		world1 = hpcgame::World3D(size);
		world1.generate(0.5);
	} else {
		world1 = hpcgame::World3D(input);
	}
	world2 = hpcgame::World3D(world1.get_size());

	// Run simulation
	auto start = std::clock();
	hpcgame::update_world(world1, world2, iterations);
	auto end = std::clock();

	// Print time
	std::cout << "Time: " << (end - start) / (double) CLOCKS_PER_SEC << std::endl;

	// Save to file
	world2.save(output);
	return 0;
}
