#ifndef GAUSSIAN_H
#define GAUSSIAN_H 
#include <array>
#include <vector>
#include <cstdint>

// Structure representing a single 3D Gaussian for splatting
struct Gaussian3D {
    std::array<float, 3> xyz;           // x, y, z coordinates
    std::array<float, 3> scale;         // Standard deviations along each axis
    std::array<float, 4> rotation;      // Quaternion (x, y, z, w)
    std::array<float, 3> normalxyz;     // normalized
    float opacity;                      // Alpha value
    std::array<float, 3> f_dc;          // Mean (can be same as position)
    std::vector<float> f_rest;    // Covariance matrix (symmetric, 6 unique values)
};

// Container for loaded Gaussians
struct GaussianGroup {
    std::vector<Gaussian3D> gaussians;
};

#endif // GAUSSIAN_H