#include <iostream>

#include <rtac_base/types/SonarPing.h>
using namespace rtac;

int main()
{
    Ping2D<float> p0(Linspace<float>(0.0f, 10.0f, 512),
                     HostVector<float>::linspace(-0.25*3.14, 0.25*3.14, 512));
    std::cout << p0 << std::endl;

    return 0;
}

