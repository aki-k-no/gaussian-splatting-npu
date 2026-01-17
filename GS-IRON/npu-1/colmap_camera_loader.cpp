#include "colmap_camera_loader.hpp"
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <map>

#include "camera.hpp"
#include "util.hpp"

struct CameraModel {
    uint32_t model_id;
    std::string model_name;
    int num_params;
};
const std::unordered_map<uint32_t, CameraModel> CAMERA_MODELS = {
    {0, CameraModel{0, "SIMPLE_PINHOLE", 3}},
    {1, CameraModel{1, "PINHOLE", 4}},
    {2, CameraModel{2, "SIMPLE_RADIAL", 4}},
    {3, CameraModel{3, "RADIAL", 5}},
    {4, CameraModel{4, "OPENCV", 8}},
    {5, CameraModel{5, "OPENCV_FISHEYE", 8}},
    {6, CameraModel{6, "FULL_OPENCV", 12}},
    {7, CameraModel{7, "FOV", 5}},
    {8, CameraModel{8, "SIMPLE_RADIAL_FISHEYE", 4}},
    {9, CameraModel{9, "RADIAL_FISHEYE", 5}},
    {10, CameraModel{10, "THIN_PRISM_FISHEYE", 12}}
};

struct Image {
    uint32_t id;
    Eigen::Vector4f qvec;
    Eigen::Vector3f tvec;
    uint32_t camera_id;
    std::string name;
    std::vector<Eigen::Vector2d> xys;
    std::vector<int64_t> point3D_ids;
};


struct CameraInfo {
    uint32_t id;
    std::string model;
    uint64_t width;
    uint64_t height;
    std::vector<double> params;
};

std::map<uint32_t, Image> read_extrinsics_binary(const std::string& path){
    std::map<uint32_t, Image> images;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open " + path);
    }

    uint64_t num_images = readBinary<uint64_t>(ifs);

    for (uint64_t i = 0; i < num_images; ++i) {
        Image img;

        img.id = readBinary<uint32_t>(ifs);

        float q1 = (float) readBinary<double>(ifs);
        float q2 = (float) readBinary<double>(ifs);
        float q3 = (float) readBinary<double>(ifs);
        float q4 = (float) readBinary<double>(ifs);

        img.qvec = Eigen::Vector4f(
            q1, q2, q3, q4
        );

        float t1 = (float) readBinary<double>(ifs);
        float t2 = (float) readBinary<double>(ifs);
        float t3 = (float) readBinary<double>(ifs);


        img.tvec = Eigen::Vector3f(
            t1, t2, t3
        );

        img.camera_id = readBinary<uint32_t>(ifs);
        img.name = readCString(ifs);

        uint64_t num_points2D = readBinary<uint64_t>(ifs);
        img.xys.reserve(num_points2D);
        img.point3D_ids.reserve(num_points2D);

        for (uint64_t j = 0; j < num_points2D; ++j) {
            double x = readBinary<double>(ifs);
            double y = readBinary<double>(ifs);
            int64_t pid = readBinary<int64_t>(ifs);
            img.xys.emplace_back(x, y);
            img.point3D_ids.push_back(pid);
        }
        images[img.id] = std::move(img);
    }

    return images;
}


std::unordered_map<uint32_t, CameraInfo> read_intrinsics_binary(const std::string& path){
    std::unordered_map<uint32_t, CameraInfo> cameras;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open camera file: " + path);
    }

    uint64_t num_cameras = readBinary<uint64_t>(ifs);

    for (uint64_t i = 0; i < num_cameras; ++i) {

        uint32_t camera_id   = readBinary<uint32_t>(ifs);
        uint32_t model_id    = readBinary<uint32_t>(ifs);
        uint64_t width       = readBinary<uint64_t>(ifs);
        uint64_t height      = readBinary<uint64_t>(ifs);

        auto it = CAMERA_MODELS.find(model_id);
        if (it == CAMERA_MODELS.end()) {
            throw std::runtime_error("Unknown camera model id: " + std::to_string(model_id));
        }

        const CameraModel& model = it->second;

        std::vector<double> params(model.num_params);
        for (int i = 0; i < model.num_params; ++i) {
            params[i] = readBinary<double>(ifs);
        }

        CameraInfo cam;
        cam.id = camera_id;
        cam.model = model.model_name;
        cam.width = width;
        cam.height = height;
        cam.params = std::move(params);

        cameras[camera_id] = std::move(cam);
    }

    return cameras;
}

void readColmapCameras(
    const std::map<uint32_t, Image>& cam_extrinsics,
    const std::unordered_map<uint32_t, CameraInfo>& cam_intrinsics,
    std::vector<Camera> &cam_infos)
{
    size_t idx = 0;

    cam_infos.reserve(cam_extrinsics.size());

    for (const auto& [id, extr] : cam_extrinsics) {

        std::cout << "\rReading camera " << ++idx << "/" << cam_extrinsics.size() << std::flush;

        const auto& intr = cam_intrinsics.at(extr.camera_id);

        Camera cam;
        cam.R = qvec2rotmat(extr.qvec);
        cam.T = extr.tvec;

        // ---- intrinsics ----
        if (intr.model == "SIMPLE_PINHOLE") {
            double f = intr.params[0];
            cam.fovY = focal2fov(f, intr.height);
            cam.fovX = focal2fov(f, intr.width);
        }
        else if (intr.model == "PINHOLE") {
            double fx = intr.params[0];
            double fy = intr.params[1];
            cam.fovX = focal2fov(fx, intr.width);
            cam.fovY = focal2fov(fy, intr.height);
        }
        else {
            throw std::runtime_error("Unsupported camera model: " + intr.model);
        }


        cam.width = intr.width > 1600 ? 1600 : intr.width;
        cam.height = intr.height * cam.width / intr.width;

        setup_camera(cam);
        cam.fx = fov2focal(cam.fovX, cam.width);
        cam.fy = fov2focal(cam.fovY, cam.height);

        cam_infos.push_back(cam);
    }

    std::cout << std::endl;
}

void loadColmapCameras(const std::string& colmap_path, std::vector<Camera> &cameras){
    auto cam_extrinsics = read_extrinsics_binary(colmap_path + "/images.bin");
    auto cam_intrinsics = read_intrinsics_binary(colmap_path + "/cameras.bin");

    readColmapCameras(cam_extrinsics, cam_intrinsics, cameras);
}
