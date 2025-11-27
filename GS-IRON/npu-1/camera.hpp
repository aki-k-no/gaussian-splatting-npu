#ifndef CAMERA_H
#define CAMERA_H
#include <Eigen/Dense>

struct Camera {
    Eigen::Matrix3f R; // rotation matrix
    Eigen::Vector3f T; // translation vector
    Eigen::Vector3f pos; // camera position
    Eigen::Matrix4f full_proj;
    Eigen::Matrix4f world_to_view;
    float fx;       // Focal length in x
    float fy;       // Focal length in y
    float cx;       // Principal point x
    float cy;       // Principal point y
    int width;      // Image width
    int height;     // Image height
};

#endif // CAMERA_H