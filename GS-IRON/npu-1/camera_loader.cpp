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
    cam.fx = fov2focal(cam.fovX, cam.width);
    cam.fovY = focal2fov(cam.fx, cam.height);
    cam.fy = fov2focal(cam.fovY, cam.height);


    setup_camera(cam);


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
    bufInA[54] = cam.pos[0];
    bufInA[55] = cam.pos[1];
    bufInA[56] = cam.pos[2];
    bufInA[57] = 0;
    bufInA[58] = 0;
    bufInA[59] = 0;
    bufInA[60] = 0;
    bufInA[61] = 0;
    #endif
    
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