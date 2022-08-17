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
    
    #ifdef RTAC_KERNEL 
    __device__ T operator()(float u, float v) const { return tex2D<T>(handle, u, v); }
    #endif //RTAC_KERNEL 
};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_VIEW_2D_HCU_
