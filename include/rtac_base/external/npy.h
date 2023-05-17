#ifndef _DEF_RTAC_BASE_EXTERNAL_NPY_H_
#define _DEF_RTAC_BASE_EXTERNAL_NPY_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdint>

#include <rtac_base/Exception.h>
#include <rtac_base/types/TypeInfo.h>

namespace rtac { namespace external {

template <typename T> inline
void save_npy_array(const std::string& filename,
                    const std::vector<std::size_t>& dims,
                    std::size_t dataSize, const T* data)
{
    std::ofstream f(filename, std::ofstream::binary);
    if(!f.is_open()) {
        throw FileError() << " : could not open file for writing '"
                          << filename << '\'';
    }

    f.write("\x93NUMPY", 6);
    f.put((char)0x1);
    f.put((char)0x0);

    std::ostringstream oss;
    oss << "{'descr':'" << GetNumpyCode<T>::value << "',"
        << "'fortran_order':False,"
        << "'shape': (";
    for(auto d : dims) {
        oss << d << ',';
    }
    oss << ")}\n";
    
    // this align the header on 16 bytes
    while((10 + oss.tellp()) & 0xf) oss << ' ';

    std::string header = oss.str();
    uint16_t headerLength = header.size();
    
    f.write((const char*)&headerLength, 2);
    f.write(header.c_str(), header.size());

    f.write((const char*)data, sizeof(T)*dataSize);
}

} //namespace external
} //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_NPY_H_
