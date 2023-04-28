#include <iostream>

#include <rtac_base/common.h>

struct S0 {
    char  x;
    float y;
};

RTAC_PACKED_STRUCT(
struct S1 {
    char  x;
    float y;
};);

int main()
{
    std::cout << "S0 : " << sizeof(S0) << std::endl;
    std::cout << "S1 : " << sizeof(S1) << std::endl;

    return 0;
}



