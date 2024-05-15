#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N1> <N2> <N3>\n";
        return 1;
    }

    int64_t N1 = std::stoll(argv[1]);
    int64_t N2 = std::stoll(argv[2]);
    int64_t N3 = std::stoll(argv[3]);

    std::ofstream file("matrix_data.bin", std::ios::binary);

    // Write the dimensions to the file
    file.write(reinterpret_cast<char*>(&N1), sizeof(N1));
    file.write(reinterpret_cast<char*>(&N2), sizeof(N2));
    file.write(reinterpret_cast<char*>(&N3), sizeof(N3));

    // Initialize random number generator
    std::srand(std::time(0));

    // Create and write the first matrix
    std::vector<double> M1(N1 * N2);
    for (auto& element : M1) {
        element = static_cast<double>(std::rand()) / RAND_MAX;  // Random double between 0 and 1
    }
    file.write(reinterpret_cast<char*>(M1.data()), M1.size() * sizeof(double));

    // Create and write the second matrix
    std::vector<double> M2(N2 * N3);
    for (auto& element : M2) {
        element = static_cast<double>(std::rand()) / RAND_MAX;  // Random double between 0 and 1
    }
    file.write(reinterpret_cast<char*>(M2.data()), M2.size() * sizeof(double));

    file.close();

    return 0;
}