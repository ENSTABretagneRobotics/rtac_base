#include <iostream>

#include <rtac_base/containers/HostVector.h>
using namespace rtac;

#include <rtac_base/external/npy.h>
using namespace external;

int main()
{
    auto data = HostVector<float>::linspace(-1.0f, 1.0, 1024);
    save_npy_array("output.npy", {32,32}, data.size(), data.data());

    return 0;
}
