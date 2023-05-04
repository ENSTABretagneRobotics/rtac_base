#include <iostream>

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/types/SonarPing.h>
#include <rtac_base/serialization/sonar_ping.h>
using namespace rtac;

int main()
{
    auto bearings0 = HostVector<float>::linspace(-10.0,10.0,32);
    Ping2D<float,HostVector> ping0(Linspace<float>(0.0,20.0,64),
                                   bearings0);
    for(unsigned int r = 0; r < ping0.range_count(); r++) {
        for(unsigned int b = 0; b < ping0.bearing_count(); b++) {
            ping0(r,b) = (r + b) & 0x1;
        }
    }
    std::cout << ping0 << std::endl;

    std::ostringstream oss;
    serialize(oss, ping0);

    Ping2D<float,HostVector> ping1;
    std::istringstream iss(oss.str());
    deserialize(iss, ping1);
    std::cout << ping1 << std::endl;
    for(unsigned int r = 0; r < ping1.range_count(); r++) {
        for(unsigned int b = 0; b < ping1.bearing_count(); b++) {
            std::cout << ' ' << ping1(r,b);
        }
        std::cout << std::endl;
    }

    return 0;
}
