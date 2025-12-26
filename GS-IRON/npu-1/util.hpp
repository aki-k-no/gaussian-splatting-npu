#ifndef UTIL_H
#define UTIL_H  
#include <array>
#include <Eigen/Dense>
#include <stdfloat>
#include <iostream>

#include "camera.hpp"

void getRelatedTiles(const std::array<float, 2>& screen_coord, int max_radius,std::array<int, 2>& min_bound, std::array<int, 2>& max_bound, std::array<int, 2>& grid_bound);
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

inline float get_float_from_pointer(void* ptr) {
    float value;
    float* p = reinterpret_cast<float*>(ptr);
    value = p[0];
    return value;
}


inline float focal2fov(float focal, float pixels) {
    return 2.0 * std::atan(pixels / (2.0 * focal));
}

inline float fov2focal(float fov, float pixels) {
    return pixels / (2.0 * std::tan(fov / 2.0));
}
inline void setup_camera(Camera& cam){

    // compute intrinsic parameters
    cam.cx = cam.width / 2.0f;
    cam.cy = cam.height / 2.0f;
    cam.tan_fovX = tan(cam.fovX / 2);
    cam.tan_fovY = tan(cam.fovY / 2);
    // put matrix
    cam.world_to_view.block<3,3>(0,0) = cam.R;
    cam.world_to_view.block<3,1>(0,3) = cam.T;
    cam.world_to_view.block<1,1>(3,3) <<  1.f;
    cam.world_to_view(3,0) = 0.f;
    cam.world_to_view(3,1) = 0.f;
    cam.world_to_view(3,2) = 0.f;

    Eigen::Matrix4f proj_mat;
    proj_mat = getProjMat(100,0.01,0.69,0.69);

    cam.full_proj = proj_mat * cam.world_to_view;
    cam.pos = cam.world_to_view.transpose().inverse().block<1,3>(3,0);
}

inline Eigen::Matrix3f qvec2rotmat(const Eigen::Vector4f& qvec) {
    Eigen::Matrix3f R;
    R(0, 0) = 1 - 2 * qvec[2] * qvec[2] - 2 * qvec[3] * qvec[3];
    R(0, 1) = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3];
    R(0, 2) = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2];
    R(1, 0) = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3];
    R(1, 1) = 1 - 2 * qvec[1] * qvec[1] - 2 * qvec[3] * qvec[3];
    R(1, 2) = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1];
    R(2, 0) = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2];
    R(2, 1) = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1];
    R(2, 2) = 1 - 2 * qvec[1] * qvec[1] - 2 * qvec[2] * qvec[2];
    return R;
}

#endif // UTIL_H