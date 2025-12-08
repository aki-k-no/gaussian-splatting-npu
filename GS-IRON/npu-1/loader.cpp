#include "gaussian.hpp"
#include "base.hpp"
#include "util.hpp"
#include "const.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>

GaussianGroup loadGaussiansFromFile(const std::string &filename) {
    // Gaussians
    GaussianGroup group;

    // open file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    bool isLittleEndian = true;
    std::vector<std::string> propertyNames;
    std::vector<std::string> propertyTypes;

    
    int count;

    // load from file
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if(token == "ply"){
            // always in front of header
            continue;
        }else if(token == "end_header"){
            //end of header
            break;
        }else if(token == "format"){
            // 
            std::string formatType;
            int version;
            iss >> formatType >> version;
            isLittleEndian = (formatType=="binary_little_endian");
            std::cout << "Format Type: " << formatType << ", Version: " << version << "\n";
        }else if(token == "element"){
            // number of elements
            std::string elementType;
            iss >> elementType >> count;
            std::cout << "Element Type: " << elementType << ", Count: " << count << "\n";
        }
        else if(token == "property"){
            // property definition
            std::string propertyType, propertyName;
            iss >> propertyType >> propertyName;
            propertyNames.push_back(propertyName);
            propertyTypes.push_back(propertyType);
            //std::cout << "Property Type: " << propertyType << ", Name: " << propertyName << "\n";

            }
    }

    group.xyz_buf = new std::bfloat16_t[((count - 1)/ CHUNK_SIZE + 1) * CHUNK_SIZE * 15];
    memset(group.xyz_buf, 0, (((count / CHUNK_SIZE - 1) + 1) * CHUNK_SIZE * 15) * sizeof(std::bfloat16_t));
    int size = ((count / CHUNK_SIZE - 1) + 1) * CHUNK_SIZE * 15;

    // Read Gaussian data
    for(int i=0;i<count;i++){

        int chunk_id = i / CHUNK_SIZE;
        int chunk_offset = i % CHUNK_SIZE;
        
        int tile_id = chunk_offset / TILE_SIZE;
        int tile_itr = chunk_offset % TILE_SIZE;
        int loop_itr = tile_itr / 4 * 32;
        int loop_res = tile_itr % 4;
        int loop_id = chunk_id * CHUNK_SIZE * 15 + TILE_SIZE * tile_id * 15 + loop_itr + loop_res;

        int tile_id2 = tile_itr / (TILE_SIZE / CONV3D_TILE_NUM);
        int tile_itr2 = tile_itr % (TILE_SIZE / CONV3D_TILE_NUM);
        int rot_id = chunk_id * CHUNK_SIZE * 15 + TILE_SIZE * tile_id * 15 + TILE_SIZE * 8 + tile_id2 * (TILE_SIZE / CONV3D_TILE_NUM) * 7 + tile_itr2 / 16 * 64 + tile_itr2 % 16;
        int scale_id = chunk_id * CHUNK_SIZE * 15 + TILE_SIZE * tile_id * 15 + TILE_SIZE * 8 + tile_id2 * (TILE_SIZE / CONV3D_TILE_NUM) * 7 + (TILE_SIZE / CONV3D_TILE_NUM) * 4 + tile_itr2 / 16 * 48 + tile_itr2 % 16;

        
        #ifdef __USE_NPU
        group.xyz_buf[loop_id + 12] = 1;
        #endif


        Gaussian3D gaussian;
        for(size_t j=0;j<propertyNames.size() ;j++){
           
            const std::string &propName = propertyNames[j];
            const std::string &propType = propertyTypes[j];
            float value;
            if(propType == "float" || propType == "float32"){
                file.read(reinterpret_cast<char*>(&value), sizeof(float));
            }else{
                throw std::runtime_error("Unsupported property type: " + propType);
            }
            // set property to gaussian
            if(propName == "x"){
                #ifdef __USE_NPU
                group.xyz_buf[loop_id] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.xyz[0] = value;
            }else if(propName == "y"){
                
                #ifdef __USE_NPU
                group.xyz_buf[loop_id + 4] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.xyz[1] = value;
                

            }else if(propName == "z"){
                
                #ifdef __USE_NPU
                group.xyz_buf[loop_id + 8] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.xyz[2] = value;
            //note that these scalings are log-based
            }else if(propName == "scale_0"){
                #ifdef __USE_NPU
                group.xyz_buf[scale_id] = float_to_bfloat16(std::exp(value));
                #else
                #endif
                gaussian.scale[0] = std::exp(value);
            }else if(propName == "scale_1"){
                #ifdef __USE_NPU
                group.xyz_buf[scale_id + 16] = float_to_bfloat16(std::exp(value));
                #else
                #endif
                gaussian.scale[1] = std::exp(value);
            }else if(propName == "scale_2"){
                #ifdef __USE_NPU
                group.xyz_buf[scale_id + 32] = float_to_bfloat16(std::exp(value));
                #else
                #endif
                gaussian.scale[2] = std::exp(value);
            }else if(propName == "rot_0"){
                #ifdef __USE_NPU
                group.xyz_buf[rot_id] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.rotation[0] = value;
            }else if(propName == "rot_1"){
                #ifdef __USE_NPU
                group.xyz_buf[rot_id + 16] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.rotation[1] = value;
            }else if(propName == "rot_2"){
                #ifdef __USE_NPU
                group.xyz_buf[rot_id + 32] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.rotation[2] = value;
            }else if(propName == "rot_3"){
                
                #ifdef __USE_NPU
                group.xyz_buf[rot_id + 48] = float_to_bfloat16(value);
                #else
                #endif
                gaussian.rotation[3] = value;
            }else if(propName == "nx"){
                gaussian.normalxyz[0] = value;
            }else if(propName == "ny"){
                gaussian.normalxyz[1] = value;
            }else if(propName == "nz"){
                gaussian.normalxyz[2] = value;
            }else if(propName == "opacity"){
                gaussian.opacity = 1 / (1 + std::exp(-value)); //sigmoid
            }else if(propName == "f_dc_0"){
                gaussian.f_dc[0] = value;
            }else if(propName == "f_dc_1"){
                gaussian.f_dc[1] = value;
            }else if(propName == "f_dc_2"){
                gaussian.f_dc[2] = value;
            }else if(propName.rfind("f_rest_", 0) == 0){
                int idx = std::stoi(propName.substr(7));
                gaussian.f_rest[idx] = value;
            }
        }
        gaussian.color = Eigen::Vector3f(0.0f, 0.0f, 0.0f); // Initialize color
        group.gaussians.push_back(gaussian);
    }
    

    return group;
}

