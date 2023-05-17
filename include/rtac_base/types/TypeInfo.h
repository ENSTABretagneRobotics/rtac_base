#ifndef _DEF_RTAC_BASE_TYPE_INFO_H_
#define _DEF_RTAC_BASE_TYPE_INFO_H_

#include <rtac_base/types/Complex.h>
#include <rtac_base/Exception.h>

namespace rtac {

struct TypeError : public Exception
{
    TypeError() : Exception("RTAC_TYPE_ERROR") {}
};

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
    RTAC_INT64,
    RTAC_CFLOAT,
    RTAC_CDOUBLE,
    RTAC_CUINT8,
    RTAC_CUINT16,
    RTAC_CUINT32,
    RTAC_CUINT64,
    RTAC_CINT8,
    RTAC_CINT16,
    RTAC_CINT32,
    RTAC_CINT64
};

template <ScalarId Id> struct ScalarTypeError : std::false_type {};
template <ScalarId Id> struct ScalarType {
    static_assert(ScalarTypeError<Id>::value, "Type not supported as scalar type in RTAC framework.");
};
template<> struct ScalarType<RTAC_FLOAT>   { using type = float;    };
template<> struct ScalarType<RTAC_DOUBLE>  { using type = double;   };
template<> struct ScalarType<RTAC_UINT8>   { using type = uint8_t;  };
template<> struct ScalarType<RTAC_UINT16>  { using type = uint16_t; };
template<> struct ScalarType<RTAC_UINT32>  { using type = uint32_t; };
template<> struct ScalarType<RTAC_UINT64>  { using type = uint64_t; };
template<> struct ScalarType<RTAC_INT8>    { using type = int8_t;   };
template<> struct ScalarType<RTAC_INT16>   { using type = int16_t;  };
template<> struct ScalarType<RTAC_INT32>   { using type = int32_t;  };
template<> struct ScalarType<RTAC_INT64>   { using type = int64_t;  };
template<> struct ScalarType<RTAC_CFLOAT>  { using type = Complex<float>;    };
template<> struct ScalarType<RTAC_CDOUBLE> { using type = Complex<double>;   };
template<> struct ScalarType<RTAC_CUINT8>  { using type = Complex<uint8_t>;  };
template<> struct ScalarType<RTAC_CUINT16> { using type = Complex<uint16_t>; };
template<> struct ScalarType<RTAC_CUINT32> { using type = Complex<uint32_t>; };
template<> struct ScalarType<RTAC_CUINT64> { using type = Complex<uint64_t>; };
template<> struct ScalarType<RTAC_CINT8>   { using type = Complex<int8_t>;   };
template<> struct ScalarType<RTAC_CINT16>  { using type = Complex<int16_t>;  };
template<> struct ScalarType<RTAC_CINT32>  { using type = Complex<int32_t>;  };
template<> struct ScalarType<RTAC_CINT64>  { using type = Complex<int64_t>;  };

constexpr const char* to_string(ScalarId id) {
    switch(id) {
        default:            return "Unknown";
        case RTAC_FLOAT   : return "RTAC_FLOAT";
        case RTAC_DOUBLE  : return "RTAC_DOUBLE";
        case RTAC_UINT8   : return "RTAC_UINT8";
        case RTAC_UINT16  : return "RTAC_UINT16";
        case RTAC_UINT32  : return "RTAC_UINT32";
        case RTAC_UINT64  : return "RTAC_UINT64";
        case RTAC_INT8    : return "RTAC_INT8";
        case RTAC_INT16   : return "RTAC_INT16";
        case RTAC_INT32   : return "RTAC_INT32";
        case RTAC_INT64   : return "RTAC_INT64";
        case RTAC_CFLOAT  : return "RTAC_CFLOAT";
        case RTAC_CDOUBLE : return "RTAC_CDOUBLE";
        case RTAC_CUINT8  : return "RTAC_CUINT8";
        case RTAC_CUINT16 : return "RTAC_CUINT16";
        case RTAC_CUINT32 : return "RTAC_CUINT32";
        case RTAC_CUINT64 : return "RTAC_CUINT64";
        case RTAC_CINT8   : return "RTAC_CINT8";
        case RTAC_CINT16  : return "RTAC_CINT16";
        case RTAC_CINT32  : return "RTAC_CINT32";
        case RTAC_CINT64  : return "RTAC_CINT64";
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
template<> struct GetScalarId<Complex<float>>    { static constexpr ScalarId value = RTAC_CFLOAT;  };
template<> struct GetScalarId<Complex<double>>   { static constexpr ScalarId value = RTAC_CDOUBLE; };
template<> struct GetScalarId<Complex<uint8_t>>  { static constexpr ScalarId value = RTAC_CUINT8;  };
template<> struct GetScalarId<Complex<uint16_t>> { static constexpr ScalarId value = RTAC_CUINT16; };
template<> struct GetScalarId<Complex<uint32_t>> { static constexpr ScalarId value = RTAC_CUINT32; };
template<> struct GetScalarId<Complex<uint64_t>> { static constexpr ScalarId value = RTAC_CUINT64; };
template<> struct GetScalarId<Complex<int8_t>>   { static constexpr ScalarId value = RTAC_CINT8;   };
template<> struct GetScalarId<Complex<int16_t>>  { static constexpr ScalarId value = RTAC_CINT16;  };
template<> struct GetScalarId<Complex<int32_t>>  { static constexpr ScalarId value = RTAC_CINT32;  };
template<> struct GetScalarId<Complex<int64_t>>  { static constexpr ScalarId value = RTAC_CINT64;  };

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
template<> struct IsRtacScalar<Complex<float>>     { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<double>>    { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<uint8_t>>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<uint16_t>>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<uint32_t>>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<uint64_t>>  { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<int8_t>>    { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<int16_t>>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<int32_t>>   { static constexpr bool value = true; };
template<> struct IsRtacScalar<Complex<int64_t>>   { static constexpr bool value = true; };

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

template <typename T> struct ScalarAsRealError : std::false_type {};
template <typename T> struct ScalarAsReal {
    static_assert(ScalarAsRealError<T>::value, "Type is not a rtac scalar");
};
template<> struct ScalarAsReal<float>    { static constexpr ScalarId value = RTAC_FLOAT;  };
template<> struct ScalarAsReal<double>   { static constexpr ScalarId value = RTAC_DOUBLE; };
template<> struct ScalarAsReal<uint8_t>  { static constexpr ScalarId value = RTAC_UINT8;  };
template<> struct ScalarAsReal<uint16_t> { static constexpr ScalarId value = RTAC_UINT16; };
template<> struct ScalarAsReal<uint32_t> { static constexpr ScalarId value = RTAC_UINT32; };
template<> struct ScalarAsReal<uint64_t> { static constexpr ScalarId value = RTAC_UINT64; };
template<> struct ScalarAsReal<int8_t>   { static constexpr ScalarId value = RTAC_INT8;   };
template<> struct ScalarAsReal<int16_t>  { static constexpr ScalarId value = RTAC_INT16;  };
template<> struct ScalarAsReal<int32_t>  { static constexpr ScalarId value = RTAC_INT32;  };
template<> struct ScalarAsReal<int64_t>  { static constexpr ScalarId value = RTAC_INT64;  };
template<> struct ScalarAsReal<Complex<float>>    { static constexpr ScalarId value = RTAC_FLOAT;  };
template<> struct ScalarAsReal<Complex<double>>   { static constexpr ScalarId value = RTAC_DOUBLE; };
template<> struct ScalarAsReal<Complex<uint8_t>>  { static constexpr ScalarId value = RTAC_UINT8;  };
template<> struct ScalarAsReal<Complex<uint16_t>> { static constexpr ScalarId value = RTAC_UINT16; };
template<> struct ScalarAsReal<Complex<uint32_t>> { static constexpr ScalarId value = RTAC_UINT32; };
template<> struct ScalarAsReal<Complex<uint64_t>> { static constexpr ScalarId value = RTAC_UINT64; };
template<> struct ScalarAsReal<Complex<int8_t>>   { static constexpr ScalarId value = RTAC_INT8;   };
template<> struct ScalarAsReal<Complex<int16_t>>  { static constexpr ScalarId value = RTAC_INT16;  };
template<> struct ScalarAsReal<Complex<int32_t>>  { static constexpr ScalarId value = RTAC_INT32;  };
template<> struct ScalarAsReal<Complex<int64_t>>  { static constexpr ScalarId value = RTAC_INT64;  };

template <typename T>
struct GetRealScalarId { static constexpr ScalarId value = GetScalarId<typename ScalarAsReal<T>::type>::value; };

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

template <typename T> struct GetNumpyCodeError : std::false_type {};
template <typename T> struct GetNumpyCode {
    static_assert(GetNumpyCodeError<T>::value, "Numpy code not defined for this scalar type");
};
template<> struct GetNumpyCode<float>             { static constexpr char value = 'f'; };
template<> struct GetNumpyCode<double>            { static constexpr char value = 'd'; };
template<> struct GetNumpyCode<uint8_t>           { static constexpr char value = 'B'; };
template<> struct GetNumpyCode<uint16_t>          { static constexpr char value = 'H'; };
template<> struct GetNumpyCode<uint32_t>          { static constexpr char value = 'I'; };
template<> struct GetNumpyCode<uint64_t>          { static constexpr char value = 'L'; };
template<> struct GetNumpyCode<int8_t>            { static constexpr char value = 'b'; };
template<> struct GetNumpyCode<int16_t>           { static constexpr char value = 'h'; };
template<> struct GetNumpyCode<int32_t>           { static constexpr char value = 'i'; };
template<> struct GetNumpyCode<int64_t>           { static constexpr char value = 'l'; };
template<> struct GetNumpyCode<Complex<float>>    { static constexpr char value = 'F'; };
template<> struct GetNumpyCode<Complex<double>>   { static constexpr char value = 'D'; };

} //namespace rtac

#endif //_DEF_RTAC_BASE_TYPE_INFO_H_
