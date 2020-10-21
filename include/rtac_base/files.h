#ifndef _DEF_RTAC_TOOLS_FILES_H_
#define _DEF_RTAC_TOOLS_FILES_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <list>
#include <vector>

// name of environment variable pointing to data path
#define DATA_PATH_ENV_VARIABLE "RTAC_DATA"

namespace rtac { namespace files {

const char* NotFound = "Not Found";

std::string rtac_data_path();

using PathList = std::list<std::string>;
// default search in rtac_data_path
PathList find(const std::string& reString=".*", bool followSimlink=true);
PathList find(const std::string& reString, const std::string& path,
              bool followSimlink=true);

// default search in rtac_data_path
std::string find_one(const std::string& reString=".*", bool followSimlink=true);
std::string find_one(const std::string& reString, const std::string& path,
                     bool followSimlink=true);

std::string append_extension(const std::string& path, const std::string& ext);

void write_pgm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment = "");
void write_ppm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment = "");
template <typename T>
void write_pgm(const std::string& path, size_t width, size_t height, const T* data,
               T a = 255, T b = 0, const std::string& comment = "")
{
    std::vector<uint8_t> imgData(width*height);
    for(int i = 0; i < imgData.size(); i++) {
        imgData[i] = static_cast<uint8_t>(a*data[i] + b);
    }
    write_pgm(path, width, height, reinterpret_cast<const char*>(imgData.data()), comment);
}

template <typename T>
void write_ppm(const std::string& path, size_t width, size_t height, const T* data,
               T a = 255, T b = 0, const std::string& comment = "")
{
    std::vector<uint8_t> imgData(3*width*height);
    for(int i = 0; i < imgData.size(); i++) {
        imgData[i] = static_cast<uint8_t>(a*data[i] + b);
    }
    write_ppm(path, width, height, reinterpret_cast<const char*>(imgData.data()), comment);
}

}; //namespace files
}; //namespace rtac

#endif //_DEF_RTAC_TOOLS_FILES_H_
