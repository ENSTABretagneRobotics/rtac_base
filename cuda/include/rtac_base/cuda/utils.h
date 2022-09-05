#ifndef _DEF_RTAC_BASE_CUDA_UTILS_H_
#define _DEF_RTAC_BASE_CUDA_UTILS_H_

#include <iostream>
#include <sstream>
#include <cstring>
#include <complex>

#include <cuda_runtime.h>

#include <rtac_base/cuda_defines.h>

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

// Using NVIDIA thrust framework when compiling CUDA code.
#ifdef RTAC_CUDACC
    #include <thrust/complex.h>
#endif //RTAC_CUDACC

namespace rtac { namespace cuda {

// Complex number definition : using thrust::complex when in CUDA device code
// (NVCC compiler) and std::complex when in host code (CPU side). Both are
// binary compatible (a buffer can be created an initialized on host side then
// uploaded to device and the other way around seamlessly). 
#ifdef RTAC_CUDACC
    template <typename T>
    using Complex = thrust::complex<T>;
#else
    template <typename T>
    using Complex = std::complex<T>;
#endif //RTAC_CUDACC

inline void init_cuda()
{
    // CUDA will init on the first API call.
    // This below is a no-op is CUDA already initialized.
    // (This was found in NVIDIA OptiX-SDK and it seemed that CUDA must be
    // initialized before initializing OptiX, although it is not clear why).
    CUDA_CHECK( cudaFree(0) ); 
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

inline std::ostream& operator<<(std::ostream& os, const float2& v)
{
    os << "(" << v.x << " " << v.y << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const float3& v)
{
    os << "(" << v.x << " " << v.y << " " << v.z << ")";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const float4& v)
{
    os << "(" << v.x << " " << v.y << " " << v.z << " " << v.w << ")";
    return os;
}


#endif //_DEF_RTAC_BASE_CUDA_UTILS_H_
