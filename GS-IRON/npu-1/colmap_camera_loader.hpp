#ifndef COLMAP_CAMERA_LOADER_HPP
#define COLMAP_CAMERA_LOADER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <Eigen/Core>
#include <fstream>

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

#endif  // COLMAP_CAMERA_LOADER_HPP