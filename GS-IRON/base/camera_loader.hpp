#ifndef CAMERA_LOADER_HPP
#define CAMERA_LOADER_HPP

#include <Eigen/Dense>
#include "camera.hpp"

void load_camera(Camera& cam, Eigen::Matrix4f baseMat_W2C);

#endif