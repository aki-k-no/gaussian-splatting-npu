#include "util.hpp"
#include <Eigen/Dense>
#include <array>
#include "const.hpp"
#include <algorithm>
#include <iostream>

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