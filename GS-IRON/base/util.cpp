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