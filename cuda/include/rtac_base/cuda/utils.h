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

//template <typename T>
//inline T* alloc(size_t size)
//{
//    T* res;
//    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&res), sizeof(T)*size) );
//    return res;
//}
//
//template <typename T>
//inline void free(T* devPtr)
//{
//    CUDA_CHECK( cudaFree(reinterpret_cast<void*>(devPtr)) );
//}
//
//struct memcpy
//{
//    template <typename T>
//    static void device_to_host(T* dst, const T* src, size_t count)
//    {
//        CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
//                               reinterpret_cast<const void*>(src),
//                               sizeof(T)*count,
//                               cudaMemcpyDeviceToHost) );
//    }
//
//    template <typename T>
//    static void host_to_device(T* dst, const T* src, size_t count)
//    {
//        CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
//                               reinterpret_cast<const void*>(src),
//                               sizeof(T)*count,
//                               cudaMemcpyHostToDevice) );
//    }
//
//    template <typename T>
//    static void device_to_device(T* dst, const T* src, size_t count)
//    {
//        CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
//                               reinterpret_cast<const void*>(src),
//                               sizeof(T)*count,
//                               cudaMemcpyDeviceToDevice) );
//    }
//
//    template <typename T>
//    static void host_to_host(T* dst, const T* src, size_t count)
//    {
//        CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
//                               reinterpret_cast<const void*>(src),
//                               sizeof(T)*count,
//                               cudaMemcpyHostToHost) );
//    }
//
//    template <typename T>
//    static T* host_to_device(const T& src)
//    {
//        T* dst = alloc<T>(1);
//        host_to_device(dst, &src, 1);
//        return dst;
//    }
//
//    template <typename T, typename T2>
//    static T device_to_host(const T2* src)
//    {
//        T dst;
//        device_to_host(reinterpret_cast<void*>(&dst),
//                       reinterpret_cast<const void*>(src),
//                       sizeof(T));
//        return dst;
//    }
//};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_UTILS_H_
