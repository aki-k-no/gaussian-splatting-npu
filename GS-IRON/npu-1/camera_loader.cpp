#include "camera_loader.hpp"
#include "camera.hpp"
#include "util.hpp"
#include "base.hpp"

#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

void load_camera(Camera& cam, Eigen::Matrix4f baseMat_W2C, float fovX){


    Eigen::Matrix4f baseMat_C2W;
    baseMat_C2W = baseMat_W2C.inverse();

    cam.R = baseMat_C2W.block<3,3>(0,0);
    cam.T << baseMat_C2W(0,3), baseMat_C2W(1,3), baseMat_C2W(2,3);
    cam.width = 800;
    cam.height = 800;
    cam.fovX = fovX;
    cam.fx = cam.width / (2 * tan(fovX / 2));
    cam.fovY = 2 * atan(cam.height/(2*cam.fx));
    cam.fy = cam.height / (2 * tan(cam.fovY / 2));
    cam.cx = 400.0f;
    cam.cy = 400.0f;
    cam.tan_fovX = tan(cam.fovX / 2);
    cam.tan_fovY = tan(cam.fovY / 2);


    // preprocess step
    cam.world_to_view.block<3,3>(0,0) = cam.R;
    cam.world_to_view.block<3,1>(0,3) = cam.T;
    cam.world_to_view.block<1,1>(3,3) <<  1.f;
    cam.world_to_view(3,0) = 0.f;
    cam.world_to_view(3,1) = 0.f;
    cam.world_to_view(3,2) = 0.f;


    Eigen::Matrix4f proj_mat;
    proj_mat = getProjMat(100,0.01,0.69,0.69);

    cam.full_proj = proj_mat * cam.world_to_view;
    
    // set matrix for NPU computation
    #ifdef __USE_NPU
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            bufInA[i * 4 + j] = float_to_bfloat16(cam.world_to_view(i,j));
            bufInA[i * 4 + 18 + j] = float_to_bfloat16(cam.full_proj(i,j));
        }
    }
    bufInA[16] = cam.fx;
    bufInA[17] = cam.fy;
    #endif
    
    cam.pos = cam.world_to_view.transpose().inverse().block<1,3>(3,0);
}

void load_from_file(std::string path, std::vector<Eigen::Matrix4f> &rotations, float &fovX) {
    // open json file
    std::ifstream ifs(path + "/transforms_train.json");
    if (!ifs.is_open()) {
        std::cerr << "cannot open file\n";
        return;
    }

    json j;
    ifs >> j;

    // get frames
    const auto& frames = j.at("frames");

    fovX = j.at("camera_angle_x").get<float>();

    // save rotation matrices of all frames
    
    rotations.reserve(frames.size());

    for (const auto& frame : frames) {
        const auto& tm = frame.at("transform_matrix");

        Eigen::Matrix4f R;

        // pick up and put in
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                if(c==1 || c == 2){

                    R(r, c) = -1 * tm[r][c].get<float>();
                }else{

                    R(r, c) = tm[r][c].get<float>();
                }
                
            }
        }

        rotations.push_back(R);
    }

    return;
}