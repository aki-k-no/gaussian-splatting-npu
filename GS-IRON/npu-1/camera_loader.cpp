#include "camera_loader.hpp"
#include "camera.hpp"
#include "util.hpp"
#include "base.hpp"

#include <Eigen/Dense>


void load_camera(Camera& cam, Eigen::Matrix4f baseMat_W2C){


    Eigen::Matrix4f baseMat_C2W;
    baseMat_C2W = baseMat_W2C.inverse();

    cam.R = baseMat_C2W.block<3,3>(0,0);
    cam.T << baseMat_C2W(0,3), baseMat_C2W(1,3), baseMat_C2W(2,3);
    cam.fx = 1111.1f;
    cam.fy = 1111.1f;
    cam.cx = 400.0f;
    cam.cy = 400.0f;
    cam.width = 800;
    cam.height = 800;


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
        }
    }
    #endif
    
    cam.pos = cam.world_to_view.transpose().inverse().block<1,3>(3,0);
}