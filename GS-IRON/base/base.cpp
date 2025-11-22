// base implimentation for 3DGS
// to check whether my understanding is correct or not

#ifndef BASE_CPP
#define BASE_CPP

#include "loader.hpp"
#include "camera.hpp"
#include "eigen3/Eigen/Dense"
#include "const.hpp"
#include "util.hpp"
#include "tile.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <array>


#define NUM_COEFF 16
#define antialiasing true

int main(){

    // load gaussians from file
    GaussianGroup group = loadGaussiansFromFile("point_cloud.ply");
    
    // initialize camera
    Camera cam;
    cam.R << 1.0f, 0.0f, 0.0f,
                     0.0f, 1.0f, 0.0f,
                     0.0f, 0.0f, 1.0f;
    
    cam.T << 0.0f, 0.0f, 5.0f;
    cam.fx = 800.0f;
    cam.fy = 600.0f;
    cam.cx = 400.0f;
    cam.cy = 300.0f;
    cam.width = 800;
    cam.height = 600;

    // preprocess step
    Eigen::Matrix3f R_inv = cam.R.transpose();
    Eigen::Vector3f T_inv = -R_inv * cam.T;

    // determine grid size    
    std::array<int, 2> grid = {cam.width / GRID_SIZE_X, cam.height / GRID_SIZE_Y};

    // iterate over gaussians
    std::vector<Gaussian3D> &gaussians = group.gaussians;  

    int numGaussians =  gaussians.size();
    for(int i=0;i<numGaussians;i++){
        Gaussian3D &g = group.gaussians[i];
        // transform to view space
        g.xyz_view = R_inv * g.xyz + T_inv;
        // change to pinhole camera coords
        
        g.screen_coord[0] = (g.xyz_view[0] * cam.fx) / g.xyz_view[2] + cam.cx;
        g.screen_coord[1] = (g.xyz_view[1] * cam.fy) / g.xyz_view[2] + cam.cy;
        
        
    }

    // put Gaussian into tiles
    std::vector<Tile> tiles;
    for(int tx=0;tx<grid[0];tx++){
        for(int ty=0;ty<grid[1];ty++){
            Tile tile;
            tile.tile_x = tx;
            tile.tile_y = ty;
            tiles.push_back(tile);
        }
    }

    // convergence 3D & 2D
    for(int i=0;i<numGaussians;i++){
        Gaussian3D &g = group.gaussians[i];
        Eigen::Matrix3f R;
        // convert quaternion to rotation matrix
        Eigen::Vector<float, 4> Rot;
        Rot << g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3];
        Rot.normalize();
        float qw = Rot[0];  
        float qx = Rot[1];  
        float qy = Rot[2];  
        float qz = Rot[3];
        std::cout << qw << ", " << qx << ", " << qy << ", " << qz << std::endl;
        R << 1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
             2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
             2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy;
        
        Eigen::Matrix3f S;
        S << std::exp(g.scale[0]), 0.0f, 0.0f,
              0.0f, std::exp(g.scale[1]), 0.0f,
              0.0f, 0.0f, std::exp(g.scale[2]);
        Eigen::Matrix3f M;
        M = R * S;
        Eigen::Matrix3f covariance3D = M * M.transpose();
        // project to 2D covariance
        float _z = g.xyz_view[2];
        Eigen::Matrix<float, 2, 3> J;
        J << cam.fx / _z, 0.0f, -cam.fx * g.xyz_view[0] / (_z * _z),
             0.0f, cam.fy / _z, -cam.fy * g.xyz_view[1] / (_z * _z);
        Eigen::Matrix<float, 2, 3> J_R = J * cam.R;
        Eigen::Matrix2f covariance2D = J_R * covariance3D * J_R.transpose(); 
        constexpr float h_var = 0.3f;
	    float det_cov2D = covariance2D(0,0) * covariance2D(1,1) - covariance2D(1,0) * covariance2D(1,0);
    	covariance2D(0,0) += h_var;
	    covariance2D(1,1) += h_var;
    	const float det_cov_plus_h_cov = covariance2D(0,0) * covariance2D(1,1) - covariance2D(1,0) * covariance2D(1,0);
	    float h_convolution_scaling = 1.0f;  
        if(antialiasing)
	    	h_convolution_scaling = std::sqrt(std::max(0.000025f, det_cov2D / det_cov_plus_h_cov)); // max for numerical stability

        float det = det_cov_plus_h_cov;
        // g.inv_cov_2d = covariance2D.inverse();

        // filter with tile grid
        // calc eignvals
    	float b = 0.5f * (covariance2D(0,0) + covariance2D(1,1));
        // short/long diameter
	    float lambda1 = b + std::sqrt(std::max(0.1f, b * b - det));
	    float lambda2 = b - std::sqrt(std::max(0.1f, b * b - det));
	    float radius = std::ceil(3.f * std::sqrt(std::max(lambda1, lambda2)));
        // get related tiles
        std::array<float, 2> rect_min;
        std::array<float, 2> rect_max;
        getRelatedTiles(g.screen_coord, radius, rect_min, rect_max, grid);
        if ((rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1]) == 0)
		    continue; // Gaussian does not contribute to the image, skip

        // put them into tiles
        for(int tx=rect_min[0];tx<rect_max[0];tx++){
            for(int ty=rect_min[1];ty<rect_max[1];ty++){
                tiles.at(tx * grid[1] + ty).unsorted_gaussians.push_back(&g);
            }
        }
	

        Eigen::Vector3f colors;
        colors << 0.0f, 0.0f, 0.0f;
        // compute colors based on coefficients
        Eigen::Vector3f dif = g.xyz - cam.T;
        dif.normalize();
        //implementation here is referenced from forward.cu from https://github.com/graphdeco-inria/gaussian-splatting 
  		float x = dif[0];
    	float y = dif[1];
	    float z = dif[2];
		float xx = x * x, yy = y * y, zz = z * z;
		float xy = x * y, yz = y * z, xz = x * z;
        for(int i=0;i<=2;i++){
            int base_idx = i * NUM_COEFF;
            for(int j=0;j<NUM_COEFF;j++){
                if(j == 0){
                    colors[i] += SH_C0 * g.f_dc[base_idx];
                }else if(j >=1 && j <=3){
                    colors[i] += SH_C1 * (-1 * y * g.f_rest[base_idx + 1] +
                                         z * g.f_rest[base_idx + 2] -
                                         x * g.f_rest[base_idx + 3]);
                }else if(j >=4 && j <=8){
                    colors[i] += SH_C2[0] * xy * g.f_rest[base_idx + 4] +
				                SH_C2[1] * yz * g.f_rest[base_idx + 5] +
				                SH_C2[2] * (2.0f * zz - xx - yy) * g.f_rest[base_idx + 6] +
				                SH_C2[3] * xz * g.f_rest[base_idx + 7] +
				                SH_C2[4] * (xx - yy) * g.f_rest[base_idx + 8];
                }else if(j >=9){
                    colors[i] += SH_C3[0] * y * (3.0f * xx - yy) * g.f_rest[base_idx + 9] +
				            	SH_C3[1] * xy * z * g.f_rest[base_idx + 10] +
				            	SH_C3[2] * y * (4.0f * zz - xx - yy) * g.f_rest[base_idx + 11] +
				            	SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * g.f_rest[base_idx + 12] +
				            	SH_C3[4] * x * (4.0f * zz - xx - yy) * g.f_rest[base_idx + 13] +
				            	SH_C3[5] * z * (xx - yy) * g.f_rest[base_idx + 14] +
				            	SH_C3[6] * x * (xx - 3.0f * yy) * g.f_rest[base_idx + 15];
                }
            }
        }
        g.color = colors;
    }

    // sort gaussians in each tile based on depth
    for(int tx=0;tx<grid[0];tx++){
        for(int ty=0;ty<grid[1];ty++){
            Tile &tile = tiles.at(tx * grid[1] + ty);
            std::cout << "Tile (" << tx << ", " << ty << ") has " << tile.unsorted_gaussians.size() << " gaussians." << std::endl;
            // sort based on depth
            std::sort(tile.unsorted_gaussians.begin(), tile.unsorted_gaussians.end(),
                [](Gaussian3D* a, Gaussian3D* b) {
                    return a->xyz_view[2] < b->xyz_view[2];
                });
            tile.sorted_gaussians = tile.unsorted_gaussians;
        }
    }
    
    



}


#endif // BASE_CPP
