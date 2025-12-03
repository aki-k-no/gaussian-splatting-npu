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

inline std::bfloat16_t float_to_bfloat16(float f) {
    return static_cast<std::bfloat16_t>(f);
}

#endif // UTIL_H