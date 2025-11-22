#ifndef TILE_HPP
#define TILE_HPP

#include <vector>
#include "gaussian.hpp"

struct Tile{
    int tile_x;
    int tile_y;
    std::vector<Gaussian3D*> unsorted_gaussians;
    std::vector<Gaussian3D*> sorted_gaussians;
};

#endif // TILE_HPP