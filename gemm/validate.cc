#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <queue>
#include <tuple>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N1 N2\n";
        return 1;
    }

    int N1 = std::stoi(argv[1]);
    int N2 = std::stoi(argv[2]);

    std::vector<double> out(N1 * N2);
    std::vector<double> stdans(N1 * N2);

    std::ifstream out_file("out.data", std::ios::binary);
    out_file.read(reinterpret_cast<char*>(out.data()), N1 * N2 * sizeof(double));

    std::ifstream stdans_file("stdans.data", std::ios::binary);
    stdans_file.read(reinterpret_cast<char*>(stdans.data()), N1 * N2 * sizeof(double));

    double total_error = 0.0;
    int count = 0;

    using ErrorInfo = std::tuple<double, int, int>;  // Error, Row, Column
    std::priority_queue<ErrorInfo, std::vector<ErrorInfo>, std::greater<>> max_errors;

    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            double error = std::abs((out[i * N2 + j] - stdans[i * N2 + j]) / stdans[i * N2 + j]);
            if (error > 0.01) {
                std::cout << "Row: " << i << ", Column: " << j << ", Error: " << error << '\n';
            }
            total_error += error;
            ++count;

            if (max_errors.size() < 10 || error > std::get<0>(max_errors.top())) {
                if (max_errors.size() == 10) {
                    max_errors.pop();
                }
                max_errors.push(ErrorInfo(error, i, j));
            }
        }
    }

    std::cout << "Average error: " << total_error / count << '\n';

    std::cout << "Top 10 errors:\n";
    while (!max_errors.empty()) {
        auto [error, i, j] = max_errors.top();
        max_errors.pop();
        std::cout << "Row: " << i << ", Column: " << j << ", Error: " << error << '\n';
    }

    return 0;
}