#ifndef UTIL_H
#define UTIL_H  
#include <array>

void getRelatedTiles(const std::array<float, 2>& screen_coord, int max_radius,std::array<float, 2>& min_bound, std::array<float, 2>& max_bound, std::array<int, 2>& grid_bound);

#endif // UTIL_H