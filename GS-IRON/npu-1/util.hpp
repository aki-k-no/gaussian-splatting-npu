#ifndef UTIL_H
#define UTIL_H  
#include <array>
#include <Eigen/Dense>
#include <stdfloat>
#include <iostream>

void getRelatedTiles(const std::array<float, 2>& screen_coord, int max_radius,std::array<float, 2>& min_bound, std::array<float, 2>& max_bound, std::array<int, 2>& grid_bound);
Eigen::Matrix4f getProjMat(const float zfar, const float znear, const float fovX, const float fovY);

static inline float bfloat16_to_float(std::bfloat16_t bf16)
{
    return static_cast<float>(bf16);
}
#include <bit>
#include <cstdint>

// Round a float to the nearest representable bfloat16 value (round-to-nearest-even)
inline float round_to_bfloat16(float f) {
    uint32_t x = std::bit_cast<uint32_t>(f);
    uint32_t lsb = (x >> 16) & 1u;                // LSB of the bfloat16 mantissa (for ties-to-even)
    uint32_t bias = 0x7FFFu + lsb;                // 0x7FFF for nearest, +lsb for ties-to-even
    x += bias;
    x &= 0xFFFF0000u;                             // zero out the lower 16 bits
    return std::bit_cast<float>(x);
}
inline std::bfloat16_t float_to_bfloat16(float f) {
    return static_cast<std::bfloat16_t>(round_to_bfloat16(f));
}

#endif // UTIL_H