#ifndef _DEF_RTAC_BASE_TYPES_POINT_FORMAT_H_
#define _DEF_RTAC_BASE_TYPES_POINT_FORMAT_H_

#include <type_traits>
#include <cstdint>
#include <iostream>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/TypeInfo.h>

namespace rtac {

template <typename T> struct PointFormatError : std::false_type {};
template <typename T> struct PointFormat {
    static_assert(PointFormatError<T>::value, "PointFormat not defined for this type.");
};
template<> struct PointFormat<float>    { using ScalarType = float   ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<double>   { using ScalarType = double  ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uint8_t>  { using ScalarType = uint8_t ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uint16_t> { using ScalarType = uint16_t; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uint32_t> { using ScalarType = uint32_t; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<uint64_t> { using ScalarType = uint64_t; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<int8_t>   { using ScalarType = int8_t  ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<int16_t>  { using ScalarType = int16_t ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<int32_t>  { using ScalarType = int32_t ; static constexpr unsigned int Size = 1; };
template<> struct PointFormat<int64_t>  { using ScalarType = int64_t ; static constexpr unsigned int Size = 1; };


template <typename P1, typename P2>
struct PointsCompatible {
    using Scalar1 = typename PointFormat<P1>::ScalarType;
    using Scalar2 = typename PointFormat<P2>::ScalarType;
    static constexpr unsigned int Size1 = PointFormat<P1>::Size;
    static constexpr unsigned int Size2 = PointFormat<P2>::Size;

    static constexpr bool value = std::is_same<Scalar1,Scalar2>::value && Size1 == Size2;
};
#define RTAC_ASSERT_COMPATIBLE_POINTS(P1, P2) static_assert(PointsCompatible<P1,P2>::value, "Imcompatible point types");

template <typename P1, typename P2> RTAC_HOSTDEVICE constexpr
bool points_compatible(const P1&, const P2&) { return PointsCompatible<P1,P2>::value; }


struct PointInfo
{
    ScalarId scalarId;
    unsigned int size;

    PointInfo() = default;
    PointInfo(const PointInfo&) = default;

    RTAC_HOSTDEVICE constexpr PointInfo(ScalarId sid, unsigned int s) : scalarId(sid), size(s) {}
};

template <typename PointT> RTAC_HOSTDEVICE constexpr
PointInfo make_point_info() { 
    return PointInfo(GetScalarId<typename PointFormat<PointT>::ScalarType>::value, PointFormat<PointT>::Size);
}
template <typename PointT> RTAC_HOSTDEVICE constexpr
PointInfo make_point_info(const PointT&) { return make_point_info<PointT>(); }

} // namespace rtac

inline std::ostream& operator<<(std::ostream& os, const rtac::PointInfo& info) {
    os << "PointInfo (type: " << to_string(info.scalarId) 
       << ", size: " << info.size << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POINT_FORMAT_H_

