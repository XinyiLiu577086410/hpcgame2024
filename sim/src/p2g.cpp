#include <array>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <tuple>

using std::vector, std::array, std::tuple, std::string;

void particle2grid(int resolution, int numparticle,
                   const vector<double> &particle_position,
                   const vector<double> &particle_velocity,
                   vector<double> &velocityu, vector<double> &velocityv,
                   vector<double> &weightu, vector<double> &weightv) {
    auto _particle_position = &particle_position[0];
    auto _particle_velocity = &particle_velocity[0];
    auto _velocityu = &velocityu[0];
    auto _velocityv = &velocityv[0];
    auto _weightu = &weightu[0];
    auto _weightv = &weightv[0];
    double grid_spacing = 1.0 / resolution;
    double inv_grid_spacing = 1.0 / grid_spacing;
    auto get_frac = [&inv_grid_spacing](double x, double y) {
        int xidx = floor(x * inv_grid_spacing);
        int yidx = floor(y * inv_grid_spacing);
        double fracx = x * inv_grid_spacing - xidx;
        double fracy = y * inv_grid_spacing - yidx;
        return tuple(array<int, 2>{xidx, yidx}, array<double, 4>{fracx * fracy, (1 - fracx) * fracy, fracx * (1 - fracy), (1 - fracx) * (1 - fracy)});
    };
    array<int, 4> offsetx = {0, 1, 0, 1};
    array<int, 4> offsety = {0, 0, 1, 1};
    auto _offsetx = &offsetx[0];
    auto _offsety = &offsety[0];
    #pragma omp parallel for
    for (int i = 0; i < numparticle; i++) {
        auto i2 = i << 1;
        auto [idxu, fracu] = get_frac(_particle_position[i2], _particle_position[i2 + 1] - 0.5 * grid_spacing);
        auto [idxv, fracv] = get_frac(_particle_position[i2] - 0.5 * grid_spacing, _particle_position[i2 + 1]);

        for (int j = 0; j < 4; j++) {
            int tmpidx = 0;
            if(i%2){
                tmpidx = (idxu[0] + _offsetx[j]) * resolution + (idxu[1] + _offsety[j]);
                #pragma omp atomic
                _velocityu[tmpidx] += _particle_velocity[i2] * fracu[j];
                #pragma omp atomic
                _weightu[tmpidx] += fracu[j];
            
                tmpidx = (idxv[0] + _offsetx[j]) * (resolution + 1) + (idxv[1] + _offsety[j]);
                #pragma omp atomic
                _velocityv[tmpidx] += _particle_velocity[i2 + 1] * fracv[j];
                #pragma omp atomic
                _weightv[tmpidx] += fracv[j];
            } else {
                tmpidx = (idxv[0] + _offsetx[j]) * (resolution + 1) + (idxv[1] + _offsety[j]);
                #pragma omp atomic
                _velocityv[tmpidx] += _particle_velocity[i2 + 1] * fracv[j];
                #pragma omp atomic
                _weightv[tmpidx] += fracv[j];
                
                tmpidx = (idxu[0] + _offsetx[j]) * resolution + (idxu[1] + _offsety[j]);
                #pragma omp atomic
                _velocityu[tmpidx] += _particle_velocity[i2] * fracu[j];
                #pragma omp atomic
                _weightu[tmpidx] += fracu[j];
            }
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s inputfile\n", argv[0]);
        return -1;
    }

    string inputfile(argv[1]);
    std::ifstream fin(inputfile, std::ios::binary);
    if (!fin) {
        printf("Error opening file");
        return -1;
    }
    
    int resolution;
    int numparticle;
    vector<double> particle_position;
    vector<double> particle_velocity;

    fin.read((char *)(&resolution), sizeof(int));
    fin.read((char *)(&numparticle), sizeof(int));
    
    particle_position.resize(numparticle * 2);
    particle_velocity.resize(numparticle * 2);
    
    printf("resolution: %d\n", resolution);
    printf("numparticle: %d\n", numparticle);
    
    fin.read((char *)(particle_position.data()),
             sizeof(double) * particle_position.size());
    fin.read((char *)(particle_velocity.data()),
             sizeof(double) * particle_velocity.size());

    vector<double> velocityu((resolution + 1) * resolution, 0.0);
    vector<double> velocityv((resolution + 1) * resolution, 0.0);
    vector<double> weightu((resolution + 1) * resolution, 0.0);
    vector<double> weightv((resolution + 1) * resolution, 0.0);


    string outputfile;

    particle2grid(resolution, numparticle, particle_position,
                    particle_velocity, velocityu, velocityv, weightu,
                    weightv);
    outputfile = "output.dat";

    std::ofstream fout(outputfile, std::ios::binary);
    if (!fout) {
        printf("Error output file");
        return -1;
    }
    fout.write((char *)(&resolution), sizeof(int));
    fout.write(reinterpret_cast<char *>(velocityu.data()),
               sizeof(double) * velocityu.size());
    fout.write(reinterpret_cast<char *>(velocityv.data()),
               sizeof(double) * velocityv.size());
    fout.write(reinterpret_cast<char *>(weightu.data()),
               sizeof(double) * weightu.size());
    fout.write(reinterpret_cast<char *>(weightv.data()),
               sizeof(double) * weightv.size());

    return 0;
}