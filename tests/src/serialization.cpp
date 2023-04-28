#include <iostream>
#include <fstream>

#include <rtac_base/serialization/serialization.h>
using namespace rtac;

int main()
{
    std::string data("abc");

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

    return 0;
}


