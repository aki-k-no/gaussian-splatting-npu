// base implimentation for 3DGS
// to check whether my understanding is correct or not


#ifndef BASE_CPP
#define BASE_CPP


#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include "cxxopts.hpp"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <Eigen/Dense>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <fstream>

#include "opencv2/opencv.hpp"

#define NUM_COEFF 15
#define antialiasing false


#include "loader.hpp"
#include "camera.hpp"
#include "const.hpp"
#include "util.hpp"
#include "tile.hpp"
#include "camera_loader.hpp"
#include "base.hpp"


int verbosity;

std::vector<uint32_t> instr_v;
xrt::device device;
xrt::kernel kernel;
xrt::bo bo_instr;
xrt::bo bo_inA;
xrt::bo bo_inB;
xrt::bo bo_outC;
xrt::bo bo_trace;
void *bufInstr;
DATATYPE_IN1 *bufInA;
DATATYPE_IN2 *bufInB;
DATATYPE_OUT *bufOut;

unsigned int opcode = 3;

std::vector<float> deviation;

std::vector<float> deviation1;
std::vector<float> deviation2;
std::vector<float> deviation3;
std::vector<float> deviation4;
std::vector<float> deviation5;
std::vector<float> deviation6;

std::vector<float> ave_npu_time;
std::vector<float> ave_total_time;
void setup_npu(int argc, const char *argv[]){

    
    // Program arguments parsing
    cxxopts::Options options("section-3");
    test_utils::add_default_options(options);

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);
    verbosity = vm["verbosity"].as<int>();

    // Load instruction sequence
    instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    

    // Start the XRT context and load the kernel
    

    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

    // set up the buffer objects
    bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    bo_inA = xrt::bo(device, IN1_SIZE * sizeof(DATATYPE_IN1),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    bo_inB = xrt::bo(device, CHUNK_SIZE * 72 * sizeof(DATATYPE_IN2),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    bo_outC = xrt::bo(device, CHUNK_SIZE * 14 * sizeof(DATATYPE_OUT),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
                            

    bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    
    bufInA = bo_inA.map<DATATYPE_IN1 *>();
    bufInB = bo_inB.map<DATATYPE_IN2 *>();
    bufOut = bo_outC.map<DATATYPE_OUT *>();



}

int cnt = 0;
inline bool compute_cov2D(GaussianGroup &group, Gaussian3D& g, Camera &cam, std::array<int, 2> &grid, std::vector<Tile> &tiles){
    
        Eigen::Vector4f pos_vec;
        pos_vec << g.xyz[0], g.xyz[1], g.xyz[2], 1.0f;
        g.xyz_view = (cam.world_to_view * pos_vec).head<3>();
        
        Eigen::Matrix3f R;
        // convert quaternion to rotation matrix
        Eigen::Vector<float, 4> Rot;
        Rot << g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3];
        Rot.normalize();
        float qw = Rot[0];  
        float qx = Rot[1];  
        float qy = Rot[2];  
        float qz = Rot[3];
        R << 1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
             2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
             2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy;
        
        Eigen::Matrix3f S;
        S << g.scale[0], 0.0f, 0.0f,
              0.0f, g.scale[1], 0.0f,
              0.0f, 0.0f, g.scale[2];

        Eigen::Matrix3f M;
        M = R * S;
        g.covariance3D = M * M.transpose();

        //project to 2D covariance
        float _x = g.xyz_view[0];
        float _y = g.xyz_view[1];
        float _z = g.xyz_view[2];

        _x = std::min(1.3f * cam.tan_fovX, std::max(-1.3f * cam.tan_fovX, _x / _z)) * _z;
        _y = std::min(1.3f * cam.tan_fovY, std::max(-1.3f * cam.tan_fovY, _y / _z)) * _z;
        Eigen::Matrix<float, 3, 3> J;
        J << cam.fx / _z, 0.0f, -cam.fx * _x / (_z * _z),
             0.0f, cam.fy / _z, -cam.fy * _y / (_z * _z),
             0.f, 0.f, 0.f;
        g.J_R = J * cam.world_to_view.block<3,3>(0,0);

        Eigen::Matrix3f covariance2D = g.J_R * g.covariance3D * g.J_R.transpose(); 
        constexpr float h_var = 0.3f;
	    float det_cov2D = covariance2D(0,0) * covariance2D(1,1) - covariance2D(1,0) * covariance2D(1,0);
    	covariance2D(0,0) += h_var;
	    covariance2D(1,1) += h_var;
    	const float det_cov_plus_h_cov = covariance2D(0,0) * covariance2D(1,1) - covariance2D(1,0) * covariance2D(1,0);
	    float h_convolution_scaling = 1.0f;  
        if(antialiasing)
	    	h_convolution_scaling = std::sqrt(std::max(0.000025f, det_cov2D / det_cov_plus_h_cov)); // max for numerical stability
            g.opacity *= h_convolution_scaling;
        

        float det = det_cov_plus_h_cov;

        if(det == 0){
            return false;
        }
        float det_inv = 1.f / det;
        
        // we use this afterwards

        g.inv_cov_2d << covariance2D(1,1) * det_inv, -covariance2D(1,0) * det_inv,
                        -covariance2D(1,0) * det_inv, covariance2D(0,0) * det_inv;

        // filter with tile grid
        // calc eignvals
    	float b = 0.5f * (covariance2D(0,0) + covariance2D(1,1));
        // short/long diameter
	    float lambda1 = b + std::sqrt(std::max(0.1f, b * b - det));
	    //float lambda2 = b - std::sqrt(std::max(0.1f, b * b - det));
        g.radius = std::ceil(3.f * std::sqrt(lambda1));



        
        // get related tiles
        std::array<int, 2> rect_min;
        std::array<int, 2> rect_max;
        getRelatedTiles(g.screen_coord, g.radius, rect_min, rect_max, grid);
        if ((rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0)
		    return false; // Gaussian does not contribute to the image, skip

        
        
        
        

        // put them into tiles
        for(int tx=rect_min[0];tx<rect_max[0];tx++){
            for(int ty=rect_min[1];ty<rect_max[1];ty++){
                tiles.at(tx * grid[1] + ty).unsorted_gaussians.push_back(&g);
            }
        }
        return true;
}

void clear_buf(GaussianGroup &group){
    memset(group.xyz_view_buf.data(), 0, group.xyz_view_buf.size() * sizeof(std::bfloat16_t));

    memset(group.screen_coord_buf.data(), 0, group.screen_coord_buf.size() * sizeof(float));

    memset(group.cov2d_buf.data(), 0, group.cov2d_buf.size() * sizeof(std::bfloat16_t));

    memset(group.color_buf.data(), 0, group.color_buf.size() * sizeof(std::bfloat16_t));

}

int omit_count = 0;

void move_to_gaussian(std::vector<Tile> &tiles, int i, GaussianGroup &group, DATATYPE_OUT *bufOut, int numGaussians, Camera& cam, std::array<int, 2> &grid){
    // extract the data and save
    memcpy(group.xyz_view_buf.data() + i * CHUNK_SIZE * 4, bufOut, CHUNK_SIZE * 4 * sizeof(DATATYPE_IN2));
    memcpy(group.color_buf.data() + i * CHUNK_SIZE * 4, bufOut + CHUNK_SIZE * 10, CHUNK_SIZE * 4 * sizeof(DATATYPE_IN2));
    memcpy(group.cov2d_buf.data() + i * CHUNK_SIZE * 4, bufOut + CHUNK_SIZE * 6, CHUNK_SIZE * 4 * sizeof(DATATYPE_IN2));
    
    
    int load_idx = CHUNK_SIZE * 0;
    int load_idx2 = CHUNK_SIZE * 6;
    int gaussian_idx = i * CHUNK_SIZE;
    for(int i = 0; i < CHUNK_SIZE; i++){
        if(gaussian_idx >= numGaussians)
            break;

        Gaussian3D &g = group.gaussians[gaussian_idx];
        gaussian_idx++;
        //group.xyz_view_buf[g.index * 4 + 2] * (1 + 1/512.0f * g.opacity)
        g.screen_depth_index = get_float_from_pointer(bufOut + CHUNK_SIZE * 4 + i * 2) ; // add opacity for sorting faster
        
            
        if(g.screen_depth_index < 0.2f){
            // dont forget to increment load_idx2
            load_idx2 += 4;
            load_idx += 4;
            continue;
        }
        
        g.screen_coord[0] = get_float_from_pointer(bufOut + load_idx);
        g.screen_coord[1] = get_float_from_pointer(bufOut + load_idx + 2);

        if(g.index == 266399){
            std::cout << g.screen_coord[0] << ", " << g.screen_coord[1] << ", depth: " << g.screen_depth_index << std::endl;
        }

                    
        float a1 = bfloat16_to_float(bufOut[load_idx2]);
        float a2 = bfloat16_to_float(bufOut[load_idx2 + 1]);
        float a3 = bfloat16_to_float(bufOut[load_idx2 + 2]);
        g.inv_cov_2d << a1, a2,
                       a2, a3;
        load_idx2 += 4;
        load_idx += 4;
        if(std::isnan(a1) && g.opacity > 0.03f || g.scale[0] > 0.3f || g.scale[1] > 0.3f || g.scale[2] > 0.3f){
            omit_count++;
            compute_cov2D(group, g, cam, grid, tiles);
            continue;
        }
        
        
        
                
        // get related tiles
        std::array<int, 2> rect_min;
        std::array<int, 2> rect_max;
        getRelatedTiles(g.screen_coord, group.cov2d_buf[g.index * 4 + 3], rect_min, rect_max, grid);
        if ((rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0)
	        continue; // Gaussian does not contribute to the image, skip

        // put them into tiles
        for(int tx=rect_min[0] * grid[1];tx<rect_max[0] * grid[1];tx+=grid[1]){
            for(int ty=rect_min[1];ty<rect_max[1];ty++){
                //if(g.index >= 80535 && g.index <= 80535){
                    tiles.at(tx + ty).unsorted_gaussians.push_back(&g);
                
            }
        }
    }
}


void render(GaussianGroup &group, Camera &cam, std::string img_name){
    
    omit_count = 0;
    

    auto start = std::chrono::steady_clock::now();


    // iterate over gaussians
    std::vector<Gaussian3D> &gaussians = group.gaussians;  

    
    // determine grid size    
    std::array<int, 2> grid = {(cam.width + GRID_SIZE_X - 1) / GRID_SIZE_X, (cam.height + GRID_SIZE_Y - 1) / GRID_SIZE_Y};
    
    // declear tiles
    std::vector<Tile> tiles;
    for(int tx=0;tx<grid[0];tx++){
        for(int ty=0;ty<grid[1];ty++){
            Tile tile;
            tile.tile_x = tx;
            tile.tile_y = ty;
            tiles.push_back(tile);
        }
    }


    int numGaussians =  gaussians.size();

    #ifdef __USE_NPU
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // instead, calculate with NPU
    auto start_npu_all = std::chrono::steady_clock::now();
    
    uint32_t tmp = 0;

    //double buffering
    DATATYPE_OUT bufOut2[CHUNK_SIZE * 14];
    
    //copy the data first
    memcpy(bufInB, group.xyz_buf.data(), CHUNK_SIZE * 72 * sizeof(DATATYPE_IN2));
            
    auto start_npu = std::chrono::steady_clock::now();
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
    run.wait();
        
    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto end_npu = std::chrono::steady_clock::now();
    auto diff_npu = std::chrono::duration_cast<std::chrono::microseconds>(end_npu - start_npu);
    tmp += diff_npu.count();
    memcpy(bufOut2, bufOut, CHUNK_SIZE * 14 * sizeof(DATATYPE_IN2));
    
    for(int i=0; i < (numGaussians - 1) / CHUNK_SIZE + 1; i++){
        
        //copy the data unless last
        if(i != (numGaussians - 1) / CHUNK_SIZE){
        start_npu = std::chrono::steady_clock::now();
        memcpy(bufInB, group.xyz_buf.data() + (i + 1) * CHUNK_SIZE * 72, CHUNK_SIZE * 72 * sizeof(DATATYPE_IN2));
        
        bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_outC);
        }
        move_to_gaussian(tiles, i, group, bufOut2, numGaussians, cam, grid);
        if(i != (numGaussians - 1) / CHUNK_SIZE){
        run.wait();
        
        // Sync device to host memories unless last
        bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(bufOut2, bufOut, CHUNK_SIZE * 14 * sizeof(DATATYPE_IN2));
        end_npu = std::chrono::steady_clock::now();
        diff_npu = std::chrono::duration_cast<std::chrono::microseconds>(end_npu - start_npu);
        tmp += diff_npu.count();
        }
        
    }
    
    auto end_npu_all = std::chrono::steady_clock::now();
    auto diff_npu_all = std::chrono::duration_cast<std::chrono::microseconds>(end_npu_all - start_npu_all);
    std::cout << "Total NPU Elapsed" << diff_npu_all.count() << " micro sec\n";

    std::cout << "NPU itself Elapsed" << tmp << " micro sec\n";
    #endif

    #ifndef __USE_NPU
    for(int i=0;i<numGaussians;i++){
        Gaussian3D &g = group.gaussians[i];
        // transform to view space
        Eigen::Vector4f pos_vec;
        pos_vec << g.xyz[0], g.xyz[1], g.xyz[2], 1.0f;
        Eigen::Vector4f pos_view = cam.full_proj * pos_vec;
        float w = pos_view[3];
        pos_view /= w + 0.0000001f; // prevent div by zero
        g.xyz_view = (cam.world_to_view * pos_vec).head<3>();
        

        g.screen_coord[0] = ((pos_view[0] + 1.0) * cam.width - 1.0) * 0.5;
        g.screen_coord[1] = ((pos_view[1] + 1.0) * cam.height - 1.0) * 0.5;

        
        
    }


    // convergence 3D & 2D
    for(int i=0;i<numGaussians;i++){
        Gaussian3D &g = group.gaussians[i];
        
        
        if(g.xyz_view[2] < 0.2f){
            continue;
        }

        bool isSafe = compute_cov2D(group, g, cam, grid, tiles);
        if(!isSafe){
            continue;
        }
	
        Eigen::Vector3f colors;
        colors << 0.5f, 0.5f, 0.5f;
        // compute colors based on coefficients
        Eigen::Vector3f dif = g.xyz - cam.pos;
        dif = dif / dif.norm();
        //implementation here is referenced from forward.cu from https://github.com/graphdeco-inria/gaussian-splatting 
  		float x = dif[0];
    	float y = dif[1];
	    float z = dif[2];
		float xx = x * x, yy = y * y, zz = z * z;
		float xy = x * y, yz = y * z, xz = x * z;
        for(int i=0;i<=2;i++){
            int base_idx = i * NUM_COEFF;
            colors[i] += SH_C0 * g.f_dc[i];
            colors[i] += SH_C1 * (-1 * y * g.f_rest[base_idx] +
                                z * g.f_rest[base_idx + 1] -
                                x * g.f_rest[base_idx + 2]);
            colors[i] += SH_C2[0] * xy * g.f_rest[base_idx + 3] +
	                     SH_C2[1] * yz * g.f_rest[base_idx + 4] +
		                 SH_C2[2] * (2.0f * zz - xx - yy) * g.f_rest[base_idx + 5] +
		                 SH_C2[3] * xz * g.f_rest[base_idx + 6] +
		                 SH_C2[4] * (xx - yy) * g.f_rest[base_idx + 7];
            colors[i] += SH_C3[0] * y * (3.0f * xx - yy) * g.f_rest[base_idx + 8] +
		               	 SH_C3[1] * xy * z * g.f_rest[base_idx + 9] +
				         SH_C3[2] * y * (4.0f * zz - xx - yy) * g.f_rest[base_idx + 10] +
				         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * g.f_rest[base_idx + 11] +
				         SH_C3[4] * x * (4.0f * zz - xx - yy) * g.f_rest[base_idx + 12] +
				         SH_C3[5] * z * (xx - yy) * g.f_rest[base_idx + 13] +
				         SH_C3[6] * x * (xx - 3.0f * yy) * g.f_rest[base_idx + 14];
            
            //clamp
            colors[i] = std::max(0.0f, colors[i]);
        }
        g.color = colors;
    }
    #endif
    
    std::cout << "omit count due to small depth: " << omit_count << std::endl;

    auto middle = std::chrono::steady_clock::now();
    auto diff_mid = std::chrono::duration_cast<std::chrono::microseconds>(middle - start);
    std::cout << "Elapsed at middle" << diff_mid.count() << " micro sec\n";
    ave_npu_time.push_back((float)diff_mid.count());        

    // sort gaussians in each tile based on depth
    for(int tx=0;tx<grid[0];tx++){
        for(int ty=0;ty<grid[1];ty++){
            Tile &tile = tiles.at(tx * grid[1] + ty);
            
            if(tile.unsorted_gaussians.size() == 0)
                continue;
            std::sort(tile.unsorted_gaussians.begin(), tile.unsorted_gaussians.end(),
                [&group](Gaussian3D* a, Gaussian3D* b) {
                    #ifdef __USE_NPU
                        return a->screen_depth_index < b->screen_depth_index;
                    #else
                        return a->xyz_view[2] < b->xyz_view[2];
                    #endif
                });
            tile.sorted_gaussians = tile.unsorted_gaussians;
        }
    }
    #ifdef __DO_RENDER
    // rendering
    cv::Mat image(cam.height, cam.width, CV_32FC3, cv::Scalar(0,0,0));

    for(int tx=0;tx<grid[0];tx++){
        for(int ty=0;ty<grid[1];ty++){
            for(int i=0;i<GRID_SIZE_X;i++){
                for(int j=0;j<GRID_SIZE_Y;j++){
                    int pixel_x = tx * GRID_SIZE_X + i;
                    int pixel_y = ty * GRID_SIZE_Y + j;
                    // this can be happen if tx, ty is close to the boundary
                    if(pixel_x >= cam.width || pixel_y >= cam.height)
                        continue;
                    Tile &tile = tiles.at(tx * grid[1] + ty);
                    Eigen::Vector3f pixel_color;
                    pixel_color << 0.f, 0.f, 0.f;
                    float pixel_opacity = 1.0f;
                    
                    for(Gaussian3D* g_ptr : tile.sorted_gaussians){
                        Gaussian3D &g = *g_ptr;

                        // compute contribution to pixel
                        Eigen::Vector2f diff;
                        //difference to the center of gaussian
                        diff[0] = pixel_x + 0.5f - g.screen_coord[0];
                        diff[1] = pixel_y + 0.5f - g.screen_coord[1];
                       
                        
                        float exponent = -0.5f * (g.inv_cov_2d(0,0) * diff[0] * diff[0] + g.inv_cov_2d(1,1) * diff[1] * diff[1]) - g.inv_cov_2d(0,1) * diff[0] * diff[1];// diff.transpose() * g.inv_cov_2d * diff;
                        
                        if(exponent > 0.f){ 
                            continue;
                        }
                        float weight = std::exp(exponent); 
                        float alpha = std::min(0.99f, g.opacity * weight);


                        
            			if (alpha < 1.0f / 255.0f || std::isinf(weight))
			            	continue;
                        
                        


                        #ifdef __USE_NPU
                        int idx = g.index * 4;
                        float true_opacity = pixel_opacity * alpha;
                        pixel_color[0] +=  true_opacity * group.color_buf[idx];
                        pixel_color[1] +=  true_opacity * group.color_buf[idx + 1];
                        pixel_color[2] +=  true_opacity * group.color_buf[idx + 2];
                        #else
                        pixel_color += pixel_opacity * alpha * g.color;
                        #endif
                        pixel_opacity = pixel_opacity * (1.f - alpha);
                        if(pixel_opacity <= 0.0001f){
                            //early return
                            break; 
                        }
                    }
                    //clamp color
                    pixel_color[0] = std::min(1.0f, pixel_color[0]);
                    pixel_color[1] = std::min(1.0f, pixel_color[1]);
                    pixel_color[2] = std::min(1.0f, pixel_color[2]); 
                    pixel_color[0] = std::max(0.0f, pixel_color[0]);
                    pixel_color[1] = std::max(0.0f, pixel_color[1]);
                    pixel_color[2] = std::max(0.0f, pixel_color[2]);   
                           
                    // store to buffer
                    image.at<cv::Vec3f>(pixel_y, pixel_x) = cv::Vec3f(pixel_color[2], pixel_color[1], pixel_color[0]);
                }
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed" << diff.count() << " micro sec\n";
    ave_total_time.push_back((float)diff.count());

    //output to file
    cv::Mat display;
    image.convertTo(display, CV_8UC3, 255.0);
    cv::imwrite(img_name, display);
    #endif

}

void render_once(std::string name){
    std::string path = "data/" + name;


    
    // load gaussians from file
    std::string ply_name = path + "/point_cloud.ply";
    
    GaussianGroup group;
    loadGaussiansFromFile(ply_name, group);

    

    // initialize camera
    std::vector<Camera> cameras;
    load_cameras_from_file(path, cameras);


    #ifdef __USE_NPU
    path = path + "/npu";
    std::cout << "Using NPU for rendering." << std::endl;
    #else
    path = path + "/cpu";
    std::cout << "Using CPU for rendering." << std::endl;
    #endif

    //if the output directory does not exist, create it
    std::string command = "mkdir -p " + path;;
    system(command.c_str());

    std::cout << "doing dataset : " << name << std::endl;
    for(unsigned int i=0;i < cameras.size();i++){
        std::cout << "iteration : " << i << std::endl;
        #ifdef __USE_NPU
        clear_buf(group);
        set_npu_buff(cameras[i]);
        #endif
        render(group, cameras[i] , path + "/output" + std::to_string(i) + ".png");
    }

}

int main(int argc, const char *argv[]){

    #ifdef __USE_NPU
    setup_npu(argc, argv);
    #endif

    std::vector<std::string> dataset_names = {
        "mip-flower"
        // "chair",
        // "drum",
        // "ficus",
        // "hotdog",
        // "lego",
        // "materials",
        // "mic",
        // "ship"
    };

    for(const auto &name : dataset_names){
        render_once(name);
    }


    // time output
    float sum_precomp = 0.0f;
    for(const auto &v : ave_npu_time){
        sum_precomp += v;
    }
    float ave_precomp = sum_precomp / ave_npu_time.size();
    std::cout << "Average NPU time: " << ave_precomp << " micro sec" << std::endl;
    float sum_total = 0.0f;
    for(const auto &v : ave_total_time){
        sum_total += v;
    }
    float ave_total = sum_total / ave_total_time.size();
    std::cout << "Average Total time: " << ave_total << " micro sec" << std::endl;

    for(int i=0;i<4;i++){
        
        std::ofstream ofs("csv/jacobian/cov3_diff_data_" + std::to_string(i) + ".csv");
        if (ofs) {
            std::vector<float> deviation;
            switch(i){
                case 0:
                deviation = deviation1;
                break;
                case 1:
                deviation = deviation2;
                break;
                case 2:
                deviation = deviation3;
                break;
                case 3:
                deviation = deviation4;
                break;
                case 4:
                deviation = deviation5;
                break;
                case 5:
                deviation = deviation6;
                break;
            }
            for (const auto &v : deviation) {
               ofs << v << ',';
            }
        }
    }


}

#endif // BASE_CPP
