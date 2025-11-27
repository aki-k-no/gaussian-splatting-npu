#include "gaussian.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

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

    // Read Gaussian data
    for(int i=0;i<count;i++){


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
                gaussian.xyz[0] = value;
            }else if(propName == "y"){
                gaussian.xyz[1] = value;
            }else if(propName == "z"){
                gaussian.xyz[2] = value;
            //note that these scalings are log-based
            }else if(propName == "scale_0"){
                gaussian.scale[0] = value;
            }else if(propName == "scale_1"){
                gaussian.scale[1] = value;
            }else if(propName == "scale_2"){
                gaussian.scale[2] = value;
            }else if(propName == "rot_0"){
                gaussian.rotation[0] = value;
            }else if(propName == "rot_1"){
                gaussian.rotation[1] = value;
            }else if(propName == "rot_2"){
                gaussian.rotation[2] = value;
            }else if(propName == "rot_3"){
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

