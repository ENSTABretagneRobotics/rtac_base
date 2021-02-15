#ifndef _DEF_RTAC_CUDA_TEXTURE_TEST_H_
#define _DEF_RTAC_CUDA_TEXTURE_TEST_H_

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/Texture2D.h>

namespace rtac { namespace cuda {

DeviceVector<float> render_texture(size_t width, size_t height,
                                   const Texture2D<float>& texture);
DeviceVector<float4> render_texture(size_t width, size_t height,
                                    const Texture2D<float4>& texture);

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_TEXTURE_TEST_H_
