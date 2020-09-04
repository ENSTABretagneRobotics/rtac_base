#ifndef _DEF_RTAC_TOOLS_FILES_H_
#define _DEF_RTAC_TOOLS_FILES_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <list>

// name of environment variable pointing to data path
#define DATA_PATH_ENV_VARIABLE "RTAC_DATA"

namespace rtac { namespace files {


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

void write_pgm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment = "");
void write_ppm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment = "");

}; //namespace files
}; //namespace rtac

#endif //_DEF_RTAC_TOOLS_FILES_H_
