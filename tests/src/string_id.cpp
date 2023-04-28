#include <iostream>

#include <rtac_base/types/StringId.h>
using namespace rtac;

int main()
{
    StringId<4> riff("RIFF");
    StringId<8> riff2("RIFF\0ABC");

    std::cout << riff2.string() << std::endl;
    for(unsigned int i = 0; i < 8; i++) {
        std::cout << '\'' << riff2.data[i] << "' ";
    }
    std::cout << std::endl;

    std::cout << "riff              = " <<  riff              << std::endl;
    std::cout << "riff2             = " <<  riff2             << std::endl;
    std::cout << "(riff == \"RIFF\")  = " <<  (riff == "RIFF")  << std::endl;
    std::cout << "(riff == \"RIFFC\") = " <<  (riff == "RIFFC") << std::endl;
    std::cout << "(riff == riff2)   = " <<  (riff == riff2)   << std::endl;

    return 0;
}

