#ifndef CAMERA_LOADER_HPP
#define CAMERA_LOADER_HPP

#include <Eigen/Dense>
#include "camera.hpp"
#include <vector>

void load_camera(Camera& cam, Eigen::Matrix4f baseMat_W2C);

void load_from_file(std::string path, std::vector<Eigen::Matrix4f> &rotations);

#endif