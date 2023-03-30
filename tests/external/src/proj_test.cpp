#include <iostream>
using namespace std;

#include <rtac_base/external/proj.h>
using namespace rtac::external;

int main()
{
    ProjTransform transform("epsg:2154", "epsg:4326");

    auto res = transform.forward(proj_coord(144191.700, 6833250.000, 17.919, 0.0));
    std::cout << res.v[0] << ' '
              << res.v[1] << ' '
              << res.v[2] << ' '
              << res.v[3] << std::endl;
    std::cout << res.enu.e << ' '
              << res.enu.n << ' '
              << res.enu.u << std::endl;

    std::cout << transform.forward(144191.700, 6833250.000, 17.919) << std::endl;

    return 0;
}


