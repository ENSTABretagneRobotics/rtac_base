#ifndef _DEF_RTAC_BASE_CUDA_DEFINES_H_
#define _DEF_RTAC_BASE_CUDA_DEFINES_H_

#if defined(__CUDACC__) || defined(__CUDABE__)
#   define RTAC_CUDACC
#   define RTAC_HOST       __host__
#   define RTAC_DEVICE     __device__
#   define RTAC_HOSTDEVICE __host__ __device__
#   define RTAC_INLINE     __forceinline__
#else
#   define RTAC_HOST
#   define RTAC_DEVICE
#   define RTAC_HOSTDEVICE
#   define RTAC_INLINE     inline
#endif

#if defined(__CUDA_ARCH__)
#   define RTAC_KERNEL  // this is defined only for cuda kernel code
#endif

#ifndef RTAC_BLOCKSIZE
#   define RTAC_BLOCKSIZE 512
#endif

#endif //_DEF_RTAC_BASE_CUDA_DEFINES_H_
