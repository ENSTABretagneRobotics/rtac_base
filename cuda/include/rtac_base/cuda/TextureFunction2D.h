#ifndef _DEF_RTAC_BASE_CUDA_TEXTURE_FUNCTION_2D_H_
#define _DEF_RTAC_BASE_CUDA_TEXTURE_FUNCTION_2D_H_

#include <cmath>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/functions.h>
#include <rtac_base/cuda/Texture2D.h>

namespace rtac { namespace cuda {

template <typename T, class XScalerT, class YScalerT>
struct TextureFunction2D : public Function2D<TextureFunction2D<T,XScalerT,YScalerT>>
{
    static_assert(IsFunction1D<XScalerT>::value, "Scaler must be a Function1D derived type");
    static_assert(IsFunction1D<YScalerT>::value, "Scaler must be a Function1D derived type");

    using value_type = T;
    using XScaler    = XScalerT;
    using YScaler    = YScalerT;

    cudaTextureObject_t handle_; 
    XScaler             xScaler_;
    YScaler             yScaler_;

    RTAC_HOSTDEVICE Bounds<float> x_domain() const { return xScaler_.domain(); }
    RTAC_HOSTDEVICE Bounds<float> y_domain() const { return yScaler_.domain(); }
    RTAC_HOSTDEVICE auto operator()(float x, float y) const {
        #ifdef RTAC_CUDACC
            return tex2D<T>(handle_, xScaler_(x), yScaler_(y));
        #else
            return 0;
        #endif
    }
};

struct TexCoordScaler : public Function1D<TexCoordScaler>
{
    float a_, b_;

    RTAC_HOSTDEVICE Bounds<float> domain() const {
        return Bounds<float>{-b_ / a_, (1.0f - b_) / a_};
    }
    RTAC_HOSTDEVICE float operator()(float x) const { return fmaf(a_, x, b_); }

    RTAC_HOSTDEVICE static TexCoordScaler make(uint32_t size,
                                               const Bounds<float>& domain)
    {
        // size is necessary because of the way cuda handles texture wrapping.
        // fetching at u coordinates 0.0 is the same as fetching texture
        // coordinates at 1.0. This means that the true center of the last
        // texture pixel is located at 1.0 * (size - 1) / size.  Moreover, when
        // fetching a pixel, an operation equivalent to floor(u*(size - 1)) is
        // performed on the texture coordinate. To avoid gross rounding error,
        // we add a 0.5f offset in pixel coordinates (0.5f_ / size on res.b_).
        TexCoordScaler res;
        res.a_ = (size - 1) / (size * domain.length());
        res.b_ = -domain.lower * res.a_ + 0.5 / size;
        return res;
    }
};

template <typename T, class XScaler, class YScaler> inline
auto make_texture_function(const Texture2D<T>& data,
                           const XScaler& xScaler,
                           const YScaler& yScaler)
{
    TextureFunction2D<T,XScaler,YScaler> res;
    res.handle_  = data.texture();
    res.xScaler_ = xScaler;
    res.yScaler_ = yScaler;
    return res;
}

template <typename T> inline
auto make_texture_function(const Texture2D<T>& data,
                           const Bounds<float>& xDomain,
                           const Bounds<float>& yDomain)
{
    TextureFunction2D<T,TexCoordScaler, TexCoordScaler> res;
    res.handle_  = data.texture();
    res.xScaler_ = TexCoordScaler::make(data.width(),  xDomain); 
    res.yScaler_ = TexCoordScaler::make(data.height(), yDomain);
    return res;
}


} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_FUNCTION_2D_H_
