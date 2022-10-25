#ifndef _DEF_RTAC_BASE_CUDA_TEXTURE_VIEW_2D_HCU_
#define _DEF_RTAC_BASE_CUDA_TEXTURE_VIEW_2D_HCU_

#include <cuda_runtime.h>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace cuda {

template <typename T>
struct TextureView2D
{
    using value_type = T;

    uint32_t            width_;
    uint32_t            height_;
    cudaTextureObject_t handle_;
    
    #ifdef RTAC_CUDACC // RTAC_KERNEL does not seem to work.
    __device__ T operator()(float u, float v) const { return tex2D<T>(handle_, u, v); }
    __device__ T operator()(float2 uv)        const { return tex2D<T>(handle_, uv.x, uv.y); }
    __device__ T operator()(uint32_t j, uint32_t i) const {
        // in cuda texture coordinate seems to be relative to pixel center,
        // from 0.0 first pixel to (width-1)/width last pixel.
        // (1.0 coordinate if first pixel of periodic texture)
        return tex2D<T>(handle_, ((float)j) / width_, ((float)i) / height_);
    }
    #endif //RTAC_KERNEL 
};

template <typename T>
struct TextureView1D
{
    using value_type = T;

    uint32_t            width_;
    cudaTextureObject_t handle_;
    
    #ifdef RTAC_CUDACC // RTAC_KERNEL does not seem to work.
    __device__ T operator()(float x) const { return tex1D<T>(handle_, x); }
    #endif //RTAC_KERNEL 
};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_VIEW_2D_HCU_
