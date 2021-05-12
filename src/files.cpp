#include <rtac_base/files.h>

#include <cstdlib>
#include <regex>

#include <experimental/filesystem>

namespace rtac { namespace files {

namespace fs = std::experimental::filesystem;

std::string rtac_data_path()
{
    return std::string(std::getenv(DATA_PATH_ENV_VARIABLE));
}

PathList rtac_data_paths(const std::string& delimiter)
{
    PathList paths;
    auto env = rtac_data_path();
    size_t idx = env.find(delimiter);
    while(idx != std::string::npos) {
        paths.push_back(env.substr(0, idx));
        idx = env.find(delimiter);
        env.erase(0, paths.back().length() + delimiter.length());
    }
    if(env.length() > 0) {
        paths.push_back(env);
    }
    return paths;
}

PathList find(const std::string& reString, bool followSimlink)
{
    auto paths = rtac_data_paths();
    return find(reString, paths, followSimlink);
}

PathList find(const std::string& reString, const char* path, bool followSimlink)
{
    return find(reString, std::string(path), followSimlink);
}

PathList find(const std::string& reString, const std::string& path, bool followSimlink)
{
    return find(reString, PathList({path}), followSimlink);
}

PathList find(const std::string& reString, const PathList& searchPaths, bool followSimlink)
{
    PathList paths;
    
    fs::directory_options doptions = fs::directory_options::none;
    if (followSimlink)
        doptions = fs::directory_options::follow_directory_symlink;
    
    std::regex re(reString);
    for(auto& sPath : searchPaths) {
        for(auto& path : fs::recursive_directory_iterator(sPath, doptions)) {
            if(std::regex_match(path.path().string(), re))
                paths.push_back(path.path());
        }
    }
    paths.sort();

    return paths;
}

std::string find_one(const std::string& reString, bool followSimlink)
{
    auto paths = rtac_data_paths();
    return find_one(reString, paths, followSimlink);
}

std::string find_one(const std::string& reString,
                     const char* path, bool followSimlink)
{
    return find_one(reString, std::string(path), followSimlink);
}

std::string find_one(const std::string& reString,
                     const std::string& path, bool followSimlink)
{
    return find_one(reString, PathList({path}), followSimlink);
}

std::string find_one(const std::string& reString,
                     const PathList& searchPaths, bool followSimlink)
{
    fs::directory_options doptions = fs::directory_options::none;
    if (followSimlink)
        doptions = fs::directory_options::follow_directory_symlink;
    
    std::regex re(reString);
    for(auto& sPath : searchPaths) {
        for(auto& path : fs::recursive_directory_iterator(sPath, doptions)) {
            if(std::regex_match(path.path().string(), re))
                return path.path().string();
        }
    }
    return NotFound;
}

std::string append_extension(const std::string& path, const std::string& ext)
{
    fs::path res(path);
    res.replace_extension(ext);
    return res.string();
}

void write_pgm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment)
{
    std::ofstream f;
    f.open(path, std::ios::out | std::ios::binary);
    if(!f.good())
        throw std::runtime_error("Cloud not open file for pgm export : " + path);
    f << "P5\n";
    if(comment.size() > 0) {
        std::istringstream iss(comment);
        for(std::string line; std::getline(iss, line);) {
            f << "# " << line << "\n";
        }
    }
    f << width << " " << height << "\n" << 255 << "\n";
    f.write(data, width*height); 

    f.close();
}

void write_ppm(const std::string& path, size_t width, size_t height, const char* data,
               const std::string& comment)
{
    std::ofstream f;
    f.open(path, std::ios::out | std::ios::binary);
    if(!f.good())
        throw std::runtime_error("Cloud not open file for pgm export : " + path);
    f << "P6\n"; // PPM file format magic number
    if(comment.size() > 0) {
        std::istringstream iss(comment);
        for(std::string line; std::getline(iss, line);) {
            f << "# " << line << "\n";
        }
    }
    f << width << " " << height << "\n" << 255 << "\n";
    f.write(data, width*height*3);

    f.close();
}

void read_ppm(const std::string& path, size_t& width, size_t& height,
              std::vector<uint8_t>& data)
{
    std::ifstream f;
    f.open(path, std::ios::in | std::ios::binary);
    if(!f.good())
        throw std::runtime_error("Cloud not open file for pgm export : " + path);
    
    std::array<char,256> buf;
    auto count = f.read(buf.data(), 2).gcount();
    buf[count] = '\0';

    if(count != 2 || buf[0] != 'P' || (buf[1] != '3' && buf[1] != '6')) {
        throw std::runtime_error("Invalid ppm file : \"" + path + "\"");
    }
    
    unsigned int maxValue = 0;
    f >> width;
    f >> height;
    f >> maxValue;
    
    if(maxValue > 255) {
        data.resize(6*width*height);
    }
    else {
        data.resize(3*width*height);
    }

    std::cout << "Reading .ppm file (" << width << "x" << height
              << ", max value : " << maxValue << ")" << std::endl;

    if(buf[1] == '3') {
        if(maxValue > 255) {
            unsigned int tmp;
            uint16_t* dst = (uint16_t*)data.data();
            uint16_t* end = (uint16_t*)(data.data() + data.size());
            do {
                f >> tmp;
                *dst = tmp;
                dst += 1;
                count = f.gcount();
            } while(count > 0 && dst < end);
        }
        else {
            unsigned int tmp;
            uint8_t* dst = (uint8_t*)data.data();
            uint8_t* end = (uint8_t*)(data.data() + data.size());
            do {
                f >> tmp;
                *dst = tmp;
                dst += 1;
                count = f.gcount();
            } while(count > 0 && dst < end);
        }
    }
    else {
        f.get();
        auto dst = data.data();
        do {
            count = f.read((char*)dst, 256).gcount();
            dst += count;
        } while(count == 256);
    }

    f.close();
}

}; //namespace files
}; //namespace rtac

