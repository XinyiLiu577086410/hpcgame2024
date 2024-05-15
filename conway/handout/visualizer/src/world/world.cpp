#include "world.hpp"

#include <numeric>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <random>
#include <omp.h>

#include <iostream>

namespace hpcgame {

World3D::World3D() : m_size(0), m_numel(0), m_power_size(0), m_timestamp(0) {}

World3D::World3D(const size_t dim, uint32_t timestamp)
    : m_size(dim)
    , m_numel(dim * dim * dim)
    , m_power_size(dim * dim)
    , m_timestamp(timestamp)
{
    m_data = std::make_shared<uint8_t[]>(m_numel);
    if (!m_data)
    {
        throw std::bad_alloc();
    }
}

World3D::World3D(const World3D& other)
    : m_size(other.m_size)
    , m_numel(other.m_numel)
    , m_power_size(other.m_power_size)
    , m_timestamp(other.m_timestamp)
{
    m_data = other.m_data;
}

World3D::World3D(const std::string& filename){
    load(filename);
}

World3D::~World3D(){
    m_data.reset();
}   

std::shared_ptr<uint8_t[]> World3D::get_data() const{
    return m_data;
}

size_t World3D::get_size() const{
    return m_size;
}

uint32_t World3D::get_timestamp() const{
    return m_timestamp;
}
size_t World3D::get_numel() const{
    return m_numel;
}

void World3D::set_timestamp(uint32_t timestamp){
    m_timestamp = timestamp;
}


void World3D::save(const std::string& filename){
    // File format:
    // char[8] magic = "HPCGGoL"
    // uint32_t timestamp
    // uint32_t size
    // uint32_t numel = size * size * size
    // uint8_t[numel] data

    if (filename.empty())
    {
        throw std::invalid_argument("filename cannot be empty");
    }
    
    // Open file
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file)
    {
        throw std::runtime_error("could not open file");
    }

    // Write magic
    const char magic[8] = "HPCGGoL";
    fwrite(magic, sizeof(char), 8, file);

    // Write timestamp
    fwrite(&m_timestamp, sizeof(uint32_t), 1, file);

    // Write Size and numel
    fwrite(&m_size, sizeof(uint32_t), 1, file);
    fwrite(&m_numel, sizeof(uint32_t), 1, file);

    // Write data
    fwrite(m_data.get(), sizeof(uint8_t), m_numel, file);

    // Close file
    fclose(file);
}

void World3D::load(const std::string &filename)
{
    if (filename.empty())
    {
        throw std::invalid_argument("filename cannot be empty");
    }

    std::cout << "Reading World from " << filename << std::endl;

    // Open file
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file)
    {
        throw std::runtime_error("could not open file");
    }

    // Read magic
    char magic[8];
    fread(magic, sizeof(char), 8, file);
    if (strncmp(magic, "HPCGGoL", 8) != 0)
    {
        throw std::runtime_error("invalid file format");
    }

    // Read timestamp
    fread(&m_timestamp, sizeof(uint32_t), 1, file);

    // Read size and numel
    fread(&m_size, sizeof(uint32_t), 1, file);
    fread(&m_numel, sizeof(uint32_t), 1, file);

    m_power_size = m_size * m_size;

    if (m_size * m_size * m_size != m_numel)
    {
        throw std::runtime_error("invalid file format");
    }

    // Allocate memory
    m_data = std::make_shared<uint8_t[]>(m_numel);
    if (!m_data)
    {
        throw std::bad_alloc();
    }

    // Read data
    fread(m_data.get(), sizeof(uint8_t), m_numel, file);

    // Close file
    fclose(file);
}

bool World3D::get_element(const size_t x, const size_t y, const size_t z) const{
    if (x >= m_size || y >= m_size || z >= m_size)
    {
        throw std::out_of_range("index out of range");
    }
    return m_data[x * m_power_size + y * m_size + z] != 0;
}

void World3D::generate(float density){
    if (density < 0.0f || density > 1.0f)
    {
        throw std::invalid_argument("density must be in [0, 1]");
    }

    const size_t numel = get_numel();
    auto data = get_data();

    // each thread will hold one random number generator
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0f, 1.0f);

        // Initialize world
        #pragma omp for
        for (size_t i = 0; i < numel; i++)
        {
            data[i] = dis(gen) < density ? 1 : 0;
        }
    }

    set_timestamp(0);
}

void World3D::print() const{
    for (size_t x = 0; x < m_size; x++)
    {
        for (size_t y = 0; y < m_size; y++)
        {
            for (size_t z = 0; z < m_size; z++)
            {
                std::cout << (int)get_element(x, y, z);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
    
}