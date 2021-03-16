#ifndef _DEF_RTAC_BASE_CUDA_OPERATORS_H_
#define _DEF_RTAC_BASE_CUDA_OPERATORS_H_

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

template <typename T>
struct Addition {
    static constexpr T Neutral = 0;
    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout += in; }
};

template <typename T>
struct Substraction {
    static constexpr T Neutral = 0;
    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout -= in; }
};

template <typename T>
struct Multiplication {
    static constexpr T Neutral = 1;
    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout *= in; }
};

template <typename T>
struct Division {
    static constexpr T Neutral = 1;
    RTAC_HOSTDEVICE static void apply(T& inout, const T& in)   { inout /= in; }
};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_OPERATORS_H_
