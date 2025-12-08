#ifndef GAUSSIAN_H
#define GAUSSIAN_H 
#include <array>
#include <vector>
#include <cstdint>
#include <Eigen/Dense>
#include <stdfloat>
#include <type_traits>

// Structure representing a single 3D Gaussian for splatting
struct Gaussian3D {
    Eigen::Vector3f xyz;           // x, y, z coordinates
    Eigen::Vector3f scale;         // Standard deviations along each axis
    std::array<float, 4> rotation; // Quaternion (x, y, z, w)
    Eigen::Vector3f normalxyz;     // normalized
    float opacity;                 // Alpha value
    std::array<float, 3> f_dc;     // DC 
    std::array<float, 45> f_rest;  // Rest DC
    Eigen::Vector3f xyz_view;      // xyz in view space
    std::array<float, 2> screen_coord; // screen coordinates
    Eigen::Vector3f color;         // RGB color
    Eigen::Matrix2f inv_cov_2d;    // Inverse of 2D covariance matrix
    Eigen::Matrix3f covariance3D;  // 3D covariance matrix
};

// Container for loaded Gaussians
struct GaussianGroup {
    std::vector<Gaussian3D> gaussians;
    std::bfloat16_t* xyz_buf;
};

#endif // GAUSSIAN_H