#ifndef _DEF_RTAC_BASE_CUDA_TEXTURE_UTILS_H_
#define _DEF_RTAC_BASE_CUDA_TEXTURE_UTILS_H_

#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/Image.h>

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/Texture2D.h>

namespace rtac { namespace cuda {

// These are function that can only be compiled from NVCC but must be usable by
// another compiler. Templates cannot work here so explicit instanciation must
// be made.
void render_texture(const Texture2D<float>&  tex, rtac::ImageView<float>  out);
void render_texture(const Texture2D<float2>& tex, rtac::ImageView<float2> out);
void render_texture(const Texture2D<float4>& tex, rtac::ImageView<float4> out);


template <typename T>
void render_texture(const Texture2D<T>& tex, rtac::Image<T,DeviceVector>& out)
{
    out.resize({tex.width(),tex.height()});
    render_texture(tex, out.view());
}

template <typename T>
rtac::Image<T,DeviceVector> render_texture(const Texture2D<T>& tex)
{
    rtac::Image<T,DeviceVector> out;
    render_texture(tex, out);
    return out;
}

} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_UTILS_H_
