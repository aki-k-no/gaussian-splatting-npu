#ifndef CAMERA_H
#define CAMERA_H
#include <array>
#include <eigen3/Eigen/Dense>

struct Camera {
    Eigen::Matrix3f R; // rotation matrix
    Eigen::Vector3f T; // translation vector
    float fx;       // Focal length in x
    float fy;       // Focal length in y
    float cx;       // Principal point x
    float cy;       // Principal point y
    int width;      // Image width
    int height;     // Image height
};

#endif // CAMERA_H