#include <iostream>
using namespace std;

#include <rtac_base/types/PointFormat.h>
#include <rtac_base/types/Point.h>
using namespace rtac;

int main()
{
    std::cout << "Id to string : "      << std::endl;
    std::cout << to_string(RTAC_FLOAT)  << std::endl;
    std::cout << to_string(RTAC_DOUBLE) << std::endl;
    std::cout << to_string(RTAC_UINT8)  << std::endl;
    std::cout << to_string(RTAC_UINT16) << std::endl;
    std::cout << to_string(RTAC_UINT32) << std::endl;
    std::cout << to_string(RTAC_UINT64) << std::endl;
    std::cout << to_string(RTAC_INT8)   << std::endl;
    std::cout << to_string(RTAC_INT16)  << std::endl;
    std::cout << to_string(RTAC_INT32)  << std::endl;
    std::cout << to_string(RTAC_INT64)  << std::endl;

    std::cout << "Integer Id : " << std::endl;
    std::cout << "IntInfo<1>  : " << to_string(IntInfo<1>::id) << std::endl;
    std::cout << "IntInfo<2>  : " << to_string(IntInfo<2>::id) << std::endl;
    std::cout << "IntInfo<4>  : " << to_string(IntInfo<4>::id) << std::endl;
    std::cout << "IntInfo<8>  : " << to_string(IntInfo<8>::id) << std::endl;
    std::cout << "UIntInfo<1> : " << to_string(UIntInfo<1>::id) << std::endl;
    std::cout << "UIntInfo<2> : " << to_string(UIntInfo<2>::id) << std::endl;
    std::cout << "UIntInfo<4> : " << to_string(UIntInfo<4>::id) << std::endl;
    std::cout << "UIntInfo<8> : " << to_string(UIntInfo<8>::id) << std::endl;

    std::cout << "Get scalar Id : " << std::endl;
    std::cout << "GetScalarId<float>    : " << to_string(GetScalarId<float>::value)    << std::endl;
    std::cout << "GetScalarId<double>   : " << to_string(GetScalarId<double>::value)   << std::endl;
    std::cout << "GetScalarId<uint8_t>  : " << to_string(GetScalarId<uint8_t>::value)  << std::endl;
    std::cout << "GetScalarId<uint16_t> : " << to_string(GetScalarId<uint16_t>::value) << std::endl;
    std::cout << "GetScalarId<uint32_t> : " << to_string(GetScalarId<uint32_t>::value) << std::endl;
    std::cout << "GetScalarId<uint64_t> : " << to_string(GetScalarId<uint64_t>::value) << std::endl;
    std::cout << "GetScalarId<int8_t>   : " << to_string(GetScalarId<int8_t>::value)   << std::endl;
    std::cout << "GetScalarId<int16_t>  : " << to_string(GetScalarId<int16_t>::value)  << std::endl;
    std::cout << "GetScalarId<int32_t>  : " << to_string(GetScalarId<int32_t>::value)  << std::endl;
    std::cout << "GetScalarId<int64_t>  : " << to_string(GetScalarId<int64_t>::value)  << std::endl;

    std::cout << "PointFormat<Point2<float>>          : " << make_point_info<Point2<float>>() << std::endl;
    std::cout << "PointFormat<Point3<int>>            : " << make_point_info<Point3<int>>() << std::endl;
    std::cout << "PointFormat<Point4<unsigned short>> : " << make_point_info<Point4<unsigned short>>() << std::endl;

    return 0;
}
