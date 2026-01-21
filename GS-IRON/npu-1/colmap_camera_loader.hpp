#ifndef COLMAP_CAMERA_LOADER_HPP
#define COLMAP_CAMERA_LOADER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include <fstream>

#include "camera.hpp"

template <typename T>
inline T readBinary(std::ifstream& ifs) {
    T value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

inline std::string readCString(std::ifstream& ifs) {
    std::string s;
    char c;
    while (ifs.get(c)) {
        if (c == '\0') break;
        s.push_back(c);
    }
    return s;
}

inline int extract_index(const std::string& filename)
{
    // 拡張子を除去
    auto pos = filename.find_last_of('.');
    std::string number = filename.substr(0, pos);

    return std::stoi(number);  // "00010" -> 10
}

void loadColmapCameras(const std::string& colmap_path, std::vector<Camera> &cameras);

#endif  // COLMAP_CAMERA_LOADER_HPP