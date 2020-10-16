#include <rtac_base/ply_files.h>

namespace rtac { namespace ply {

happly::PLYData New()
{
    return happly::PLYData();
}

happly::Element& new_element(happly::PLYData& data, const std::string& name, bool overwrite)
{
    if(data.hasElement(name)) {
        if(!overwrite) {
            throw std::runtime_error(".ply data already has a "
                + name + " element, and overwrite is disabled.");
        }
    }
    else {
        data.addElement(name, 1);
    }
    return data.getElement(name);
}

void write(const std::string& path, happly::PLYData& data, bool ascii)
{
    std::ofstream f(path, std::ios::out | std::ios::binary);
    if(!f.is_open()) {
        throw std::runtime_error(
            "rtac::ply::write : could not open file for writing " + path);
    }
    write(f, data, ascii);
}

void write(std::ostream& os, happly::PLYData& data, bool ascii)
{
    if(ascii)
        data.write(os, happly::DataFormat::ASCII);
    else
        data.write(os, happly::DataFormat::Binary);
}

happly::PLYData read(const std::string& path)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if(!f.is_open()) {
        throw std::runtime_error(
            "rtac::ply::write : could not open file for reading " + path);
    }
    return read(f);
}

happly::PLYData read(std::istream& is)
{
    return happly::PLYData(is);
}

}; //namespace ply
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, happly::PLYData& data)
{
    rtac::ply::write(os, data, true);
    return os;
}



