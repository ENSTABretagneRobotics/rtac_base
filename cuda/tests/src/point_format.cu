#include <iostream>

#include <rtac_base/cuda/CUDAPointFormat.h>
using namespace rtac;

int main()
{
    std::cout << make_point_info<float1>()  << std::endl;
    std::cout << make_point_info<double1>() << std::endl;
    std::cout << make_point_info<uchar1>()  << std::endl;
    std::cout << make_point_info<ushort1>() << std::endl;
    std::cout << make_point_info<uint1>()   << std::endl;
    std::cout << make_point_info<ulong1>()  << std::endl;
    std::cout << make_point_info<char1>()   << std::endl;
    std::cout << make_point_info<short1>()  << std::endl;
    std::cout << make_point_info<int1>()    << std::endl;
    std::cout << make_point_info<long1>()   << std::endl;
    std::cout << std::endl;

    std::cout << make_point_info<float2>()  << std::endl;
    std::cout << make_point_info<double2>() << std::endl;
    std::cout << make_point_info<uchar2>()  << std::endl;
    std::cout << make_point_info<ushort2>() << std::endl;
    std::cout << make_point_info<uint2>()   << std::endl;
    std::cout << make_point_info<ulong2>()  << std::endl;
    std::cout << make_point_info<char2>()   << std::endl;
    std::cout << make_point_info<short2>()  << std::endl;
    std::cout << make_point_info<int2>()    << std::endl;
    std::cout << make_point_info<long2>()   << std::endl;
    std::cout << std::endl;

    std::cout << make_point_info<float3>()  << std::endl;
    std::cout << make_point_info<double3>() << std::endl;
    std::cout << make_point_info<uchar3>()  << std::endl;
    std::cout << make_point_info<ushort3>() << std::endl;
    std::cout << make_point_info<uint3>()   << std::endl;
    std::cout << make_point_info<ulong3>()  << std::endl;
    std::cout << make_point_info<char3>()   << std::endl;
    std::cout << make_point_info<short3>()  << std::endl;
    std::cout << make_point_info<int3>()    << std::endl;
    std::cout << make_point_info<long3>()   << std::endl;
    std::cout << std::endl;

    std::cout << make_point_info<float4>()  << std::endl;
    std::cout << make_point_info<double4>() << std::endl;
    std::cout << make_point_info<uchar4>()  << std::endl;
    std::cout << make_point_info<ushort4>() << std::endl;
    std::cout << make_point_info<uint4>()   << std::endl;
    std::cout << make_point_info<ulong4>()  << std::endl;
    std::cout << make_point_info<char4>()   << std::endl;
    std::cout << make_point_info<short4>()  << std::endl;
    std::cout << make_point_info<int4>()    << std::endl;
    std::cout << make_point_info<long4>()   << std::endl;
    std::cout << std::endl;

    return 0;
}
