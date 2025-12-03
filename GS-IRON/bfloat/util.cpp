#include "util.hpp"
#include <Eigen/Dense>
#include <array>
#include "const.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

void getRelatedTiles(const std::array<float, 2>& screen_coord, int max_radius,std::array<float, 2>& min_bound, std::array<float, 2>& max_bound, std::array<int, 2>& grid_bound){
	//clamp
    min_bound[0] = std::min(grid_bound[0], std::max((int)0, (int)((screen_coord[0] - max_radius) / GRID_SIZE_X)));
	min_bound[1] = std::min(grid_bound[1], std::max((int)0, (int)((screen_coord[1] - max_radius) / GRID_SIZE_Y)));
        
	
	max_bound[0] = std::min(grid_bound[0], std::max((int)0, (int)((screen_coord[0] + max_radius + GRID_SIZE_X - 1) / GRID_SIZE_X)));
	max_bound[1] = std::min(grid_bound[1], std::max((int)0, (int)((screen_coord[1] + max_radius + GRID_SIZE_Y - 1) / GRID_SIZE_Y)));
	
}


Eigen::Matrix4f getProjMat(const float zfar, const float znear, const float fovX, const float fovY){
	float tanHalfFovY = std::tan((fovY / 2));
    float tanHalfFovX = std::tan((fovX / 2));

    float top = tanHalfFovY * znear;
    float bottom = -top;
    float right = tanHalfFovX * znear;
    float left = -right;

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();

    float z_sign = 1.0;

    P(0, 0) = 2.0 * znear / (right - left);
    P(1, 1) = 2.0 * znear / (top - bottom);
    P(0, 2) = (right + left) / (right - left);
    P(1, 2) = (top + bottom) / (top - bottom);
    P(3, 2) = z_sign;
    P(2, 2) = z_sign * zfar / (zfar - znear);
    P(2, 3) = -(zfar * znear) / (zfar - znear);
    return P;
}

float float_to_bfloat16_scale(float value){
    static_assert(sizeof(float) == 4, "float_to_bfloat16_scale assumes 32-bit float");
    uint32_t as_int;
    std::memcpy(&as_int, &value, sizeof(as_int));
    // // round to nearest-even bfloat16
    // uint32_t rounding_bias = ((as_int >> 16) & 1u) + 0x7FFFu;
    // as_int += rounding_bias;
    // keep top 16 bits, zero lower 16 bits
    uint32_t bf32 = as_int & 0xFFFF0000u;
    float result_f;
    std::memcpy(&result_f, &bf32, sizeof(result_f));
    return result_f;
}

Eigen::Vector4f float_to_bfloat_vec4f(Eigen::Vector4f value){
    Eigen::Vector4f result;
    for(int i=0;i<4;i++){
        result[i] = float_to_bfloat16_scale(value[i]);
    }
    return result;
}

Eigen::Vector3f float_to_bfloat_vec3f(Eigen::Vector3f value){
    Eigen::Vector3f result;
    for(int i=0;i<3;i++){
        result[i] = float_to_bfloat16_scale(value[i]);
    }
    return result;
}


Eigen::Matrix4f float_to_bfloat_mat4f(Eigen::Matrix4f value){
    Eigen::Matrix4f result;
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            result(i,j) = float_to_bfloat16_scale(value(i,j));
        }
    }
    return result;
}

Eigen::Matrix3f float_to_bfloat_mat3f(Eigen::Matrix3f value){
    Eigen::Matrix3f result;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            result(i,j) = float_to_bfloat16_scale(value(i,j));
        }
    }
    return result;
}