#ifndef _DEF_RTAC_BASE_TYPES_POINT_FORMAT_H_
#define _DEF_RTAC_BASE_TYPES_POINT_FORMAT_H_

#include <type_traits>
#include <cstdint>
#include <iostream>

#include <rtac_base/cuda_defines.h>

namespace rtac {

enum ScalarId {
    RTAC_FLOAT,
    RTAC_DOUBLE,
    RTAC_UINT8,
    RTAC_UINT16,
    RTAC_UINT32,
    RTAC_UINT64,
    RTAC_INT8,
    RTAC_INT16,
    RTAC_INT32,
    RTAC_INT64
};

template <ScalarId Id> struct ScalarTypeError : std::false_type {};
template <ScalarId Id> struct ScalarType {
    static_assert(ScalarTypeError<Id>::value, "Type not supported as scalar type in RTAC framework.");
};

template<> struct ScalarType<RTAC_FLOAT>  { using type = float;    };
template<> struct ScalarType<RTAC_DOUBLE> { using type = double;   };
template<> struct ScalarType<RTAC_UINT8>  { using type = uint8_t;  };
template<> struct ScalarType<RTAC_UINT16> { using type = uint16_t; };
template<> struct ScalarType<RTAC_UINT32> { using type = uint32_t; };
template<> struct ScalarType<RTAC_UINT64> { using type = uint64_t; };
template<> struct ScalarType<RTAC_INT8>   { using type = int8_t;   };
template<> struct ScalarType<RTAC_INT16>  { using type = int16_t;  };
template<> struct ScalarType<RTAC_INT32>  { using type = int32_t;  };
template<> struct ScalarType<RTAC_INT64>  { using type = int64_t;  };

constexpr const char* to_string(ScalarId id) {
    switch(id) {
        default:           return "Unknown";
        case RTAC_FLOAT  : return "RTAC_FLOAT";
        case RTAC_DOUBLE : return "RTAC_DOUBLE";
        case RTAC_UINT8  : return "RTAC_UINT8";
        case RTAC_UINT16 : return "RTAC_UINT16";
        case RTAC_UINT32 : return "RTAC_UINT32";
        case RTAC_UINT64 : return "RTAC_UINT64";
        case RTAC_INT8   : return "RTAC_INT8";
        case RTAC_INT16  : return "RTAC_INT16";
        case RTAC_INT32  : return "RTAC_INT32";
        case RTAC_INT64  : return "RTAC_INT64";
    }
}

template <typename T> struct GetScalarIdError : std::false_type {};
template <typename T> struct GetScalarId {
    static_assert(GetScalarIdError<T>::value, "Id not defined for scalar type");
};
template<> struct GetScalarId<float>    { static constexpr ScalarId value = RTAC_FLOAT;  };
template<> struct GetScalarId<double>   { static constexpr ScalarId value = RTAC_DOUBLE; };
template<> struct GetScalarId<uint8_t>  { static constexpr ScalarId value = RTAC_UINT8;  };
template<> struct GetScalarId<uint16_t> { static constexpr ScalarId value = RTAC_UINT16; };
template<> struct GetScalarId<uint32_t> { static constexpr ScalarId value = RTAC_UINT32; };
template<> struct GetScalarId<uint64_t> { static constexpr ScalarId value = RTAC_UINT64; };
template<> struct GetScalarId<int8_t>   { static constexpr ScalarId value = RTAC_INT8;   };
template<> struct GetScalarId<int16_t>  { static constexpr ScalarId value = RTAC_INT16;  };
template<> struct GetScalarId<int32_t>  { static constexpr ScalarId value = RTAC_INT32;  };
template<> struct GetScalarId<int64_t>  { static constexpr ScalarId value = RTAC_INT64;  };

template <unsigned int Size> struct IntInfoError : std::false_type {};
template <unsigned int Size> struct IntInfo { 
    static_assert(IntInfoError<Size>::value, "Unsupported integer size");
};
template<> struct IntInfo<1> { static constexpr ScalarId id = RTAC_INT8;  using type = typename ScalarType<id>::type; };
template<> struct IntInfo<2> { static constexpr ScalarId id = RTAC_INT16; using type = typename ScalarType<id>::type; };
template<> struct IntInfo<4> { static constexpr ScalarId id = RTAC_INT32; using type = typename ScalarType<id>::type; };
template<> struct IntInfo<8> { static constexpr ScalarId id = RTAC_INT64; using type = typename ScalarType<id>::type; };
template <unsigned int Size> struct UIntInfo { 
    static_assert(IntInfoError<Size>::value, "Unsupported integer size");
};
template<> struct UIntInfo<1> { static constexpr ScalarId id = RTAC_UINT8;  using type = typename ScalarType<id>::type; };
template<> struct UIntInfo<2> { static constexpr ScalarId id = RTAC_UINT16; using type = typename ScalarType<id>::type; };
template<> struct UIntInfo<4> { static constexpr ScalarId id = RTAC_UINT32; using type = typename ScalarType<id>::type; };
template<> struct UIntInfo<8> { static constexpr ScalarId id = RTAC_UINT64; using type = typename ScalarType<id>::type; };

// this does not work (multiple definition of...)
//template<> struct GetScalarId<unsigned short> { static constexpr ScalarId value = UIntInfo<sizeof(unsigned short)>::value; }
//template<> struct GetScalarId<unsigned int>   { static constexpr ScalarId value = UIntInfo<sizeof(unsigned int)>::value; }
//template<> struct GetScalarId<unsigned long>  { static constexpr ScalarId value = UlongId<sizeof(unsigned long)>::value; }
//template<> struct GetScalarId<short>          { static constexpr ScalarId value = UIntInfo<sizeof(short)>::value; }
//template<> struct GetScalarId<int>            { static constexpr ScalarId value = UIntInfo<sizeof(int)>::value; }
//template<> struct GetScalarId<long>           { static constexpr ScalarId value = UlongId<sizeof(long)>::value; }


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
struct PointTypesCompatible {
    using Scalar1 = typename PointFormat<P1>::ScalarType;
    using Scalar2 = typename PointFormat<P2>::ScalarType;
    static constexpr unsigned int Size1 = PointFormat<P1>::Size;
    static constexpr unsigned int Size2 = PointFormat<P2>::Size;

    static constexpr bool value = std::is_same<Scalar1,Scalar2>::value && Size1 == Size2;
};

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

