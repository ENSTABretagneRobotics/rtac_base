#ifndef _DEF_RTAC_CUDA_TEXTURE_TEST_H_
#define _DEF_RTAC_CUDA_TEXTURE_TEST_H_

#include <rtac_base/cuda/CudaVector.h>
#include <rtac_base/cuda/Texture2D.h>

namespace rtac { namespace cuda {

CudaVector<float> render_texture(size_t width, size_t height,
                                 const Texture2D<float>& texture);
CudaVector<float4> render_texture(size_t width, size_t height,
                                  const Texture2D<float4>& texture);

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_TEXTURE_TEST_H_
