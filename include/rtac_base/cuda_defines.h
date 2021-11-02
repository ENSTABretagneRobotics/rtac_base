#ifndef _DEF_RTAC_BASE_CUDA_DEFINES_H_
#define _DEF_RTAC_BASE_CUDA_DEFINES_H_

#if defined(__CUDACC__) || defined(__CUDABE__)
#   define RTAC_CUDACC
#   define RTAC_HOSTDEVICE __host__ __device__
#   define RTAC_INLINE     __forceinline__
#else
#   define RTAC_HOSTDEVICE
#   define RTAC_INLINE     inline
#endif

#endif //_DEF_RTAC_BASE_CUDA_DEFINES_H_
