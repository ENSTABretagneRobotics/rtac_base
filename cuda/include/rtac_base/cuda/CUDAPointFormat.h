#ifndef _DEF_RTAC_BASE_CUDA_POINT_FORMAT_H_
#define _DEF_RTAC_BASE_CUDA_POINT_FORMAT_H_

#include <rtac_base/types/PointFormat.h>

namespace rtac {

template<> struct PointFormat<float1>  { using ScalarType = float;  static constexpr unsigned int Size = 1; };
template<> struct PointFormat<double1> { using ScalarType = double; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uchar1>  { using ScalarType = typename UIntInfo<sizeof(uchar1)>::type;  static constexpr unsigned int Size = 1; };
template<> struct PointFormat<ushort1> { using ScalarType = typename UIntInfo<sizeof(ushort1)>::type; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uint1>   { using ScalarType = typename UIntInfo<sizeof(uint1)>::type;   static constexpr unsigned int Size = 1; };
template<> struct PointFormat<ulong1>  { using ScalarType = typename UIntInfo<sizeof(ulong1)>::type;  static constexpr unsigned int Size = 1; };
template<> struct PointFormat<char1>   { using ScalarType = typename IntInfo<sizeof(char1)>::type;    static constexpr unsigned int Size = 1; };
template<> struct PointFormat<short1>  { using ScalarType = typename IntInfo<sizeof(short1)>::type;   static constexpr unsigned int Size = 1; };
template<> struct PointFormat<int1>    { using ScalarType = typename IntInfo<sizeof(int1)>::type;     static constexpr unsigned int Size = 1; };
template<> struct PointFormat<long1>   { using ScalarType = typename IntInfo<sizeof(long1)>::type;    static constexpr unsigned int Size = 1; };

template<> struct PointFormat<float2>  { using ScalarType = float;  static constexpr unsigned int Size = 2; };
template<> struct PointFormat<double2> { using ScalarType = double; static constexpr unsigned int Size = 2; };
template<> struct PointFormat<uchar2>  { using ScalarType = typename UIntInfo<sizeof(uchar1)>::type;  static constexpr unsigned int Size = 2; };
template<> struct PointFormat<ushort2> { using ScalarType = typename UIntInfo<sizeof(ushort1)>::type; static constexpr unsigned int Size = 2; };
template<> struct PointFormat<uint2>   { using ScalarType = typename UIntInfo<sizeof(uint1)>::type;   static constexpr unsigned int Size = 2; };
template<> struct PointFormat<ulong2>  { using ScalarType = typename UIntInfo<sizeof(ulong1)>::type;  static constexpr unsigned int Size = 2; };
template<> struct PointFormat<char2>   { using ScalarType = typename IntInfo<sizeof(char1)>::type;    static constexpr unsigned int Size = 2; };
template<> struct PointFormat<short2>  { using ScalarType = typename IntInfo<sizeof(short1)>::type;   static constexpr unsigned int Size = 2; };
template<> struct PointFormat<int2>    { using ScalarType = typename IntInfo<sizeof(int1)>::type;     static constexpr unsigned int Size = 2; };
template<> struct PointFormat<long2>   { using ScalarType = typename IntInfo<sizeof(long1)>::type;    static constexpr unsigned int Size = 2; };

template<> struct PointFormat<float3>  { using ScalarType = float;  static constexpr unsigned int Size = 3; };
template<> struct PointFormat<double3> { using ScalarType = double; static constexpr unsigned int Size = 3; };
template<> struct PointFormat<uchar3>  { using ScalarType = typename UIntInfo<sizeof(uchar1)>::type;  static constexpr unsigned int Size = 3; };
template<> struct PointFormat<ushort3> { using ScalarType = typename UIntInfo<sizeof(ushort1)>::type; static constexpr unsigned int Size = 3; };
template<> struct PointFormat<uint3>   { using ScalarType = typename UIntInfo<sizeof(uint1)>::type;   static constexpr unsigned int Size = 3; };
template<> struct PointFormat<ulong3>  { using ScalarType = typename UIntInfo<sizeof(ulong1)>::type;  static constexpr unsigned int Size = 3; };
template<> struct PointFormat<char3>   { using ScalarType = typename IntInfo<sizeof(char1)>::type;    static constexpr unsigned int Size = 3; };
template<> struct PointFormat<short3>  { using ScalarType = typename IntInfo<sizeof(short1)>::type;   static constexpr unsigned int Size = 3; };
template<> struct PointFormat<int3>    { using ScalarType = typename IntInfo<sizeof(int1)>::type;     static constexpr unsigned int Size = 3; };
template<> struct PointFormat<long3>   { using ScalarType = typename IntInfo<sizeof(long1)>::type;    static constexpr unsigned int Size = 3; };

template<> struct PointFormat<float4>  { using ScalarType = float;  static constexpr unsigned int Size = 4; };
template<> struct PointFormat<double4> { using ScalarType = double; static constexpr unsigned int Size = 4; };
template<> struct PointFormat<uchar4>  { using ScalarType = typename UIntInfo<sizeof(uchar1)>::type;  static constexpr unsigned int Size = 4; };
template<> struct PointFormat<ushort4> { using ScalarType = typename UIntInfo<sizeof(ushort1)>::type; static constexpr unsigned int Size = 4; };
template<> struct PointFormat<uint4>   { using ScalarType = typename UIntInfo<sizeof(uint1)>::type;   static constexpr unsigned int Size = 4; };
template<> struct PointFormat<ulong4>  { using ScalarType = typename UIntInfo<sizeof(ulong1)>::type;  static constexpr unsigned int Size = 4; };
template<> struct PointFormat<char4>   { using ScalarType = typename IntInfo<sizeof(char1)>::type;    static constexpr unsigned int Size = 4; };
template<> struct PointFormat<short4>  { using ScalarType = typename IntInfo<sizeof(short1)>::type;   static constexpr unsigned int Size = 4; };
template<> struct PointFormat<int4>    { using ScalarType = typename IntInfo<sizeof(int1)>::type;     static constexpr unsigned int Size = 4; };
template<> struct PointFormat<long4>   { using ScalarType = typename IntInfo<sizeof(long1)>::type;    static constexpr unsigned int Size = 4; };

} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_POINT_FORMAT_H_
