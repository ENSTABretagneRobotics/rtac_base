#include <iostream>
#include <fstream>

#include <rtac_base/serialization/serialization.h>
#include <rtac_base/containers/HostVector.h>
using namespace rtac;

int main()
{
    std::string data("abc");

    std::cout << "64bit aligned serialization" << std::endl;

    std::ostringstream oss64;
    oss64 << ChunkHeader64("RTAC_DAT", data.size()) << data.c_str();
    std::cout << "serialized : '" << oss64.str() << '\'' << std::endl;
    std::cout << "stream position : " << oss64.tellp() << std::endl;
    auto serialized64 = oss64.str();
    std::cout << "binary (hex) :";
    for(unsigned int i = 0; i < oss64.tellp(); i++) {
        std::cout << ' ' << std::hex << (unsigned int)serialized64[i];
    }
    std::cout << std::dec << std::endl;

    std::istringstream iss64(oss64.str());
    auto header64 = ChunkHeader64::Empty();
    HostVector<char> data64;
    iss64 >> header64 >> data64;
    std::cout << header64.id << " :";
    for(auto v : data64) {
        std::cout << ' ' << std::hex << (unsigned int)v;
    }
    std::cout << std::endl;
    std::cout << "stream position : " << std::dec << iss64.tellg() << std::endl;

    std::cout << std::endl << "32bit aligned serialization" << std::endl;
    std::ostringstream oss32;
    oss32 << ChunkHeader32("RTAC", data.size()) << data.c_str();
    std::cout << "serialized : '" << oss32.str() << '\'' << std::endl;
    std::cout << "stream position : " << oss32.tellp() << std::endl;
    auto serialized32 = oss32.str();
    std::cout << "binary (hex) :";
    for(unsigned int i = 0; i < oss32.tellp(); i++) {
        std::cout << ' ' << std::hex << (unsigned int)serialized32[i];
    }
    std::cout << std::endl;

    std::istringstream iss32(oss32.str());
    auto header32 = ChunkHeader32::Empty();
    HostVector<char> data32;
    iss32 >> header32 >> data32;
    std::cout << header32.id << " :";
    for(auto v : data32) {
        std::cout << ' ' << std::hex << (unsigned int)v;
    }
    std::cout << std::endl;
    std::cout << "stream position : " << std::dec << iss32.tellg() << std::endl;

    return 0;
}


