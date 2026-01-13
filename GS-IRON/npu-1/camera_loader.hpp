#ifndef CAMERA_LOADER_HPP
#define CAMERA_LOADER_HPP

#include <Eigen/Dense>
#include "camera.hpp"
#include <vector>
#include <string>

void set_npu_buff(Camera& cam);

void load_camera(Camera& cam, Eigen::Matrix4f baseMat_W2C, float fovX);

void load_from_file(std::string path, std::vector<Eigen::Matrix4f> &rotations, float &fovX);

void load_cameras_from_file(std::string path, std::vector<Camera>& cameras);

#endif