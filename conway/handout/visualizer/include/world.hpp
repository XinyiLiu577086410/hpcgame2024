#pragma once

#include <vector>
#include <string>
#include <cstdint> 
#include <memory>

namespace hpcgame {

class World3D
{
protected:
    // m_data is shared_ptr, no copy when copy constructor is called
    std::shared_ptr<uint8_t[]> m_data;
    size_t m_size, m_numel, m_power_size;
    uint32_t m_timestamp;
public:
    World3D();
    World3D(const size_t m_size, uint32_t timestamp = 0);
    World3D(const World3D& other);
    World3D(const std::string& filename);
    ~World3D();
    std::shared_ptr<uint8_t[]> get_data() const;
    uint32_t get_timestamp() const;
    size_t get_size() const;
    size_t get_numel() const;
    void set_timestamp(uint32_t timestamp);
    void save(const std::string& filename);
    void load(const std::string& filename);
    bool get_element(const size_t x, const size_t y, const size_t z) const;
    void print() const;
    void generate(float density = 0.5f);
};

} // namespace hpcgame