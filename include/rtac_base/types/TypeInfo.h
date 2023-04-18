#ifndef _DEF_RTAC_BASE_TYPE_INFO_H_
#define _DEF_RTAC_BASE_TYPE_INFO_H_

#include <rtac_base/types/Complex.h>

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

template <typename T> struct IsRtacScalar { static constexpr bool value = false; };
template<> struct IsRtacScalar<float>     { static constexpr bool value = true; };
template<> struct IsRtacScalar<double>    { static constexpr bool value = true; };
template<> struct IsRtacScalar<uint8_t>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<uint16_t>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<uint32_t>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<uint64_t>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<int8_t>    { static constexpr bool value = true; };
template<> struct IsRtacScalar<int16_t>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<int32_t>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<int64_t>   { static constexpr bool value = true; };

template <typename T> struct IsRtacRealScalar { static constexpr bool value = false; };
template<> struct IsRtacRealScalar<float>     { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<double>    { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<uint8_t>   { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<uint16_t>  { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<uint32_t>  { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<uint64_t>  { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<int8_t>    { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<int16_t>   { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<int32_t>   { static constexpr bool value = true; };
template<> struct IsRtacRealScalar<int64_t>   { static constexpr bool value = true; };

template <typename T> struct IsRtacComplex         { static constexpr bool value = false; };
template<> struct IsRtacComplex<Complex<float>>    { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<double>>   { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<uint8_t>>  { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<uint16_t>> { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<uint32_t>> { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<uint64_t>> { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<int8_t>>   { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<int16_t>>  { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<int32_t>>  { static constexpr bool value = true; };
template<> struct IsRtacComplex<Complex<int64_t>>  { static constexpr bool value = true; };

template <typename T> struct ScalarTypeFromComplexError : std::false_type {};
template <typename T> struct ScalarTypeFromComplex {
    static_assert(ScalarTypeFromComplexError<T>::value, "Type is not rtac::Complex");
};
template <typename T> struct ScalarTypeFromComplex<Complex<T>> { using type = T; };

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



} //namespace rtac

#endif //_DEF_RTAC_BASE_TYPE_INFO_H_
