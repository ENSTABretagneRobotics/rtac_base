#ifndef _DEF_RTAC_BASE_CUDA_OPERATORS_H_
#define _DEF_RTAC_BASE_CUDA_OPERATORS_H_

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

//template <typename T, template <typename> class OperatorT>
//struct Operator {
//    static constexpr T Neutral = OperatorT<T>::Neutral;
//    RTAC_HOSTDEVICE static void apply(T& inout, const T& in) { 
//        OperatorT<T>::apply(inout, in);
//    }
//    RTAC_HOSTDEVICE static void apply(volatile T& inout, const volatile T& in) {
//        OperatorT<volatile T>::apply(inout, in);
//    }
//};

template <typename T>
struct Addition {
    static constexpr T Neutral = 0;
    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout += in; }
};

//template <typename T>
//struct Substraction {
//    static constexpr T Neutral = 0;
//    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout -= in; }
//    RTAC_HOSTDEVICE static T apply(const T& in0, const T& in1) { return in0 - in1; }
//};
//
//template <typename T>
//struct Multiplication {
//    static constexpr T Neutral = 1;
//    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout *= in; }
//    RTAC_HOSTDEVICE static T apply(const T& in0, const T& in1) { return in0 * in1; }
//};
//
//template <typename T>
//struct Division {
//    static constexpr T Neutral = 1;
//    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout /= in; }
//    RTAC_HOSTDEVICE static T apply(const T& in0, const T& in1) { return in0 / in1; }
//};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_OPERATORS_H_
