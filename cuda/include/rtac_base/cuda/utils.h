#ifndef _DEF_RTAC_BASE_CUDA_UTILS_H_
#define _DEF_RTAC_BASE_CUDA_UTILS_H_

#include <iostream>
#include <sstream>
#include <cstring>

#include <cuda_runtime.h>


#if defined(__CUDACC__) || defined(__CUDABE__)
#   define RTAC_CUDACC
#   define RTAC_HOSTDEVICE __host__ __device__
#   define RTAC_INLINE     __forceinline__
#else
#   define RTAC_HOSTDEVICE
#   define RTAC_INLINE     inline
#endif

#define CUDA_CHECK( call )                                                 \
    do {                                                                   \
        cudaError_t code = call;                                           \
        if(code != cudaSuccess) {                                          \
            std::ostringstream oss;                                        \
            oss << "CUDA call '" << #call << "' failed '"                  \
                << cudaGetErrorString(code) << "' (code:" << code << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                    \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while(0)                                                             \

#define CUDA_CHECK_LAST( call )                                            \
    do {                                                                   \
        cudaError_t code = cudaGetLastError();                             \
        if(code != cudaSuccess) {                                          \
            std::ostringstream oss;                                        \
            oss << "CUDA call '" << #call << "' failed '"                  \
                << cudaGetErrorString(code) << "' (code:" << code << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                    \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while(0)                                                             \

namespace rtac { namespace cuda {

inline void init_cuda()
{
    // CUDA will init on the first API call.
    // This below is a no-op is CUDA already initialized.
    // (This was found in NVIDIA OptiX-SDK and it seemed that CUDA must be
    // initialized before initializing OptiX, although it is not clear why).
    CUDA_CHECK( cudaFree(0) ); 
}

template <typename T>
T zero()
{
    T res;
    std::memset(&res,0,sizeof(T));
    return res;
}

inline void set_device(int deviceOrdinal)
{
    CUDA_CHECK( cudaSetDevice(deviceOrdinal) );
}

}; //namespace cuda
}; //namespace rtac

#ifndef RTAC_CUDACC

#include <rtac_base/types/common.h>

//template <typename T>
//float2 to_float2(const rtac::types::Vector2<T>& v)
//{
//    return float2({v[0], v[1]});
//}
//template <typename T>
//float3 to_float3(const rtac::types::Vector3<T>& v)
//{
//    return float3({v[0], v[1], v[2]});
//}
//template <typename T>
//float4 to_float4(const rtac::types::Vector4<T>& v)
//{
//    return float4({v[0], v[1], v[2], v[3]});
//}

template <typename T>
float2 to_float2(const Eigen::MatrixBase<rtac::types::Vector2<T>>& v)
{
    return float2({v[0], v[1]});
}

template <typename T>
float3 to_float3(const Eigen::MatrixBase<rtac::types::Vector3<T>>& v)
{
    return float3({v[0], v[1], v[2]});
}

template <typename T>
float4 to_float4(const Eigen::MatrixBase<rtac::types::Vector4<T>>& v)
{
    return float4({v[0], v[1], v[2], v[3]});
}

#endif //RTAC_CUDACC

#endif //_DEF_RTAC_BASE_CUDA_UTILS_H_
